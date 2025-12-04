"""
import os
import time
from tqdm import tqdm
from options.options import parse
import random
import argparse

parser = argparse.ArgumentParser(description="Script for train")
parser.add_argument('-p', '--config', type=str, default='./options/train/RSBlur.yml', help = 'Config file of prediction')
args = parser.parse_args()

opt = parse(args.config)

import torch
import torch.distributed as dist

from data.dataset_tools.datapipeline import *
from archs import *
from losses import *
from data import *
from utils.utils import create_path_models, set_random_seed
from tools.trainer import train_model
from tools.tester import eval_model

torch.autograd.set_detect_anomaly(True)

#parameters for saving model
PATH_RESUME, PATH_SAVE = create_path_models(opt['save'])

final_score = 0.

def run_model():
    # Initialize distributed environment
    # dist.init_process_group(backend='nccl')

    world_size = int(os.getenv('WORLD_SIZE', 1))
    if world_size > 1:
        dist.init_process_group(backend='nccl')
    else:
        # 单卡模式，手动设置环境变量，避免后续代码出错
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        print("Running in single-GPU mode, distributed training disabled.")



    
    # Get distributed settings
    #global_rank = dist.get_rank()
    try:
        global_rank = dist.get_rank()
    except (ValueError, RuntimeError):  # 捕获进程组未初始化的错误
        global_rank = 0

    #world_size = dist.get_world_size()
    try:
        world_size = dist.get_world_size()
    except (ValueError, RuntimeError):
        world_size = 1


    # local_rank = int(os.environ['LOCAL_RANK'])
    args = parse_args()
    if args.distributed:
        # 这里才用 torch.distributed 和 LOCAL_RANK
        import torch.distributed as dist
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        global_rank = dist.get_rank()
    else:
        print("Running in single-GPU mode, distributed training disabled.")
        local_rank = 0          # ✅ 直接设为 0
        global_rank = 0         # ✅ 主进程
    
    # Set device
    torch.cuda.set_device(0)
    
    print(f'Rank: {global_rank} of {world_size}, Local rank: {local_rank}')

    seed = opt['datasets']['seed']
    if seed is None:
        seed = random.randint(1, 10000)
        opt['datasets']['seed'] = seed
    set_random_seed(seed + global_rank)

    # DEFINE NETWORK, SCHEDULER AND OPTIMIZER
    model, macs, params = create_model(opt['network'], local_rank=local_rank, global_rank=global_rank)

    # save this stats into opt 
    opt['macs'] = macs
    opt['params'] = params
    opt['Total_GPUs'] = world_size # add the number of GPUs to the opt

    # define the optimizer
    optim, scheduler = create_optim_scheduler(opt['train'], model)

    # if resume load the weights
    model, optim, scheduler, _ = resume_model(model, optim, scheduler, path_model = PATH_RESUME,
                                                         local_rank = local_rank, global_rank = global_rank, 
                                                         resume=opt['network']['resume_training'])

    # last_epochs = start_epochs
    for step in range(opt['train']['STEPS']):
        total_steps = opt['train']['STEPS']
        if global_rank == 0: print(f'--------------- In Step {step + 1} of {total_steps} Steps ---------------')
        # LOAD THE DATALOADERS
        train_loader, test_loader, samplers = create_data(global_rank, world_size=world_size, opt = opt['datasets'], step = step)

        # create losses in this step
        all_losses = create_loss(opt['train'], step = step,local_rank=local_rank, global_rank=global_rank)
        final_score= 0
        
        if global_rank==0:
            total = opt['train']['epochs'][step]
            pbar = tqdm(total = total)
        for epoch in range(opt['train']['epochs'][step]):

            start_time = time.time()
            metrics_train = {'epoch': epoch,'final_score': final_score}
            metrics_eval = {}

            # shuffle the samplers of each loader
            shuffle_sampler(samplers, epoch)

            # train phase
            model.train()
            model, optim, metrics_train = train_model(model, optim, all_losses, train_loader, metrics_train,local_rank = local_rank)

            # eval phase
            if epoch % opt['train']['eval_freq'] == 0 or epoch == opt['train']['epochs'][step] - 1:
                model.eval()
                metrics_eval = eval_model(model, test_loader, metrics_eval, local_rank=local_rank, world_size=world_size)
                
                # print some results
                if global_rank==0:
                    print(f"Epoch {epoch + 1} of {opt['train']['epochs'][step]} took {time.time() - start_time:.3f}s\n")
                    if type(next(iter(metrics_eval.values()))) == dict:
                        for key, metric_eval in metrics_eval.items():
                            print(f" \t {key} --- PSNR: {metric_eval['valid_psnr']}, SSIM: {metric_eval['valid_ssim']}, LPIPS: {metric_eval['valid_lpips']}")
                    else:
                        print(f" \t {opt['datasets']['name']} --- PSNR: {metrics_eval['valid_psnr']}, SSIM: {metrics_eval['valid_ssim']}, LPIPS: {metrics_eval['valid_lpips']}")
                    # update progress bar
                    pbar.update(1)
            # Save the model after every epoch
                final_score = save_checkpoint(model, optim, scheduler, metrics_eval = metrics_eval, metrics_train=metrics_train, 
                                        paths = PATH_SAVE,global_rank=global_rank)

            #update scheduler
            scheduler.step()
        if global_rank == 0:
            pbar.close()

if __name__ == '__main__':
    run_model()

"""

import os
import time
import random
import argparse

import torch
import torch.distributed as dist

from tqdm import tqdm
from options.options import parse
from data.dataset_tools.datapipeline import *
from archs import *
from losses import *
from data import *
from utils.utils import create_path_models, set_random_seed
from tools.trainer import train_model
from tools.tester import eval_model

torch.autograd.set_detect_anomaly(True)

# ==================== 命令行参数解析 ====================
parser = argparse.ArgumentParser(description="Script for train")
parser.add_argument('-p', '--config', type=str, default='./options/train/RSBlur.yml', help='Config file for training')
args = parser.parse_args()

# ==================== 加载配置文件 ====================
opt = parse(args.config)

# ==================== 模型保存路径 ====================
PATH_RESUME, PATH_SAVE = create_path_models(opt['save'])
final_score = 0.

def run_model():
    """
    主训练函数，处理单卡/多卡初始化并启动训练循环。
    """
    # ========== 1. 分布式环境初始化 (单卡/多卡自适应) ==========
    # 初始化关键变量（无论单卡多卡都会用到）
    world_size = 1
    global_rank = 0
    local_rank = 0
    
    # 检查是否通过环境变量启用了分布式训练
    env_world_size = int(os.getenv('WORLD_SIZE', '1'))
    
    if env_world_size > 1 and dist.is_available():
        # ---------- 多GPU分布式模式 ----------
        try:
            dist.init_process_group(backend='nccl')
            world_size = dist.get_world_size()
            global_rank = dist.get_rank()
            local_rank = int(os.environ.get('LOCAL_RANK', '0'))
            print(f"[分布式模式] 全局进程 {global_rank} / 共 {world_size}, 本地设备 {local_rank}")
        except Exception as e:
            print(f"[警告] 分布式初始化失败，将回退到单卡模式。错误: {e}")
            # 失败后回退到单卡模式
            world_size = 1
            global_rank = 0
            local_rank = 0
    else:
        # ---------- 单GPU模式 (你的使用场景) ----------
        world_size = 1
        global_rank = 0
        local_rank = 0
        print("[单卡模式] 分布式训练已禁用。")
    
    # 明确设置环境变量，防止代码其他部分因检查而报错
    os.environ['RANK'] = str(global_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(local_rank)

     # ====== 新增：为DDP初始化一个单机“分布式”环境 ======
    if not dist.is_initialized():
        # 即使单卡，也为兼容DDP而初始化进程组
        # 使用'gloo'或'ncccl'后端，gloo在Windows上更兼容
        try:
            dist.init_process_group(backend='gloo', rank=global_rank, world_size=world_size,
                                     init_method='env://')
            print(f"[兼容层] 已初始化单机进程组以适配DDP包装器。")
        except Exception as e:
            print(f"[警告] 进程组初始化失败: {e}. 尝试继续...")
    # =============================================




    # ========== 2. 设置当前进程使用的GPU ==========
    torch.cuda.set_device(local_rank)
    print(f'进程状态: 全局排名 {global_rank} / 总进程数 {world_size}, 本地GPU编号 {local_rank}')

    # ========== 3. 设置随机种子 (保证实验可复现) ==========
    seed = opt['datasets'].get('seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['datasets']['seed'] = seed
    set_random_seed(seed + global_rank)  # 为不同进程设置不同种子

    # ========== 4. 创建模型、统计计算量参数量 ==========
    model, macs, params = create_model(opt['network'], local_rank=local_rank, global_rank=global_rank)
    opt['macs'] = macs
    opt['params'] = params
    opt['Total_GPUs'] = world_size  # 将GPU数量记录到配置中

    # ========== 5. 定义优化器和学习率调度器 ==========
    optim, scheduler = create_optim_scheduler(opt['train'], model)

    # ========== 6. 加载检查点 (如果需要恢复训练) ==========
    model, optim, scheduler, _ = resume_model(model, optim, scheduler,
                                               path_model=PATH_RESUME,
                                               local_rank=local_rank,
                                               global_rank=global_rank,
                                               resume=opt['network'].get('resume_training', False))

    # ========== 7. 主训练循环 ==========
    for step in range(opt['train']['STEPS']):
        total_steps = opt['train']['STEPS']
        if global_rank == 0:
            print(f'\n--------------- 阶段 {step + 1} / {total_steps} ---------------')

        # 7.1 为当前训练阶段创建数据加载器
        train_loader, test_loader, samplers = create_data(global_rank,
                                                           world_size=world_size,
                                                           opt=opt['datasets'],
                                                           step=step)

        # 7.2 为当前阶段创建损失函数
        all_losses = create_loss(opt['train'], step=step,
                                 local_rank=local_rank,
                                 global_rank=global_rank)

        final_score = 0  # 重置每个阶段的最佳分数

        # 7.3 进度条 (仅主进程显示)
        if global_rank == 0:
            pbar = tqdm(total=opt['train']['epochs'][step])

        # 7.4 迭代每一个Epoch
        for epoch in range(opt['train']['epochs'][step]):
            epoch_start_time = time.time()
            metrics_train = {'epoch': epoch, 'final_score': final_score}
            metrics_eval = {}

            # 在每个Epoch开始时打乱训练数据顺序
            shuffle_sampler(samplers, epoch)

            # ---------- 训练阶段 ----------
            model.train()
            model, optim, metrics_train = train_model(model, optim, all_losses,
                                                      train_loader, metrics_train,
                                                      local_rank=local_rank)

            # ---------- 评估阶段 eval phase ----------
            eval_this_epoch = (epoch % opt['train']['eval_freq'] == 0) or (epoch == opt['train']['epochs'][step] - 1)
            if eval_this_epoch:
                model.eval()
                metrics_eval = eval_model(model, test_loader, metrics_eval,
                                          local_rank=local_rank,
                                          world_size=world_size)

                # 7.5 主进程打印评估结果
                if global_rank == 0:
                    # 打印耗时
                    epoch_time = time.time() - epoch_start_time
                    print(f"Epoch {epoch + 1}/{opt['train']['epochs'][step]} 耗时: {epoch_time:.3f}s")

                    # 打印评估指标 (PSNR, SSIM, LPIPS)
                    if isinstance(next(iter(metrics_eval.values())), dict):
                        # 如果有多个评估指标集
                        for key, metric_eval in metrics_eval.items():
                            print(f"  \t{key} --- PSNR: {metric_eval['valid_psnr']:.4f}, "
                                  f"SSIM: {metric_eval['valid_ssim']:.4f}, "
                                  f"LPIPS: {metric_eval['valid_lpips']:.4f}")
                    else:
                        # 只有一个评估指标集
                        dataset_name = opt['datasets']['name']
                        print(f"  \t{dataset_name} --- PSNR: {metrics_eval['valid_psnr']:.4f}, "
                              f"SSIM: {metrics_eval['valid_ssim']:.4f}, "
                              f"LPIPS: {metrics_eval['valid_lpips']:.4f}")

                    # 更新进度条
                    pbar.update(1)

                # 7.6 主进程保存检查点
                if global_rank == 0:
                    final_score = save_checkpoint(model, optim, scheduler,
                                                  metrics_eval=metrics_eval,
                                                  metrics_train=metrics_train,
                                                  paths=PATH_SAVE,
                                                  global_rank=global_rank)

            # 7.7 更新学习率
            scheduler.step()

        # 7.8 关闭当前阶段的进度条
        if global_rank == 0:
            pbar.close()

if __name__ == '__main__':
    run_model()