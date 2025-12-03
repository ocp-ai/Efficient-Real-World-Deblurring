import os
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.nn.parallel import DistributedDataParallel as DDP
from ptflops import get_model_complexity_info

from .nafnet import NAFNet, NAFNetLocal  

def create_model(opt, local_rank, global_rank=1):
    '''
    Creates the model.
    opt: a dictionary from the yaml config key network
    '''
    name = opt['name']

    if name == 'NAFNet':
        model = NAFNet(img_channel=opt['img_channels'], 
                        width=opt['width'], 
                        middle_blk_num=opt['middle_blk_num'], 
                        enc_blk_nums=opt['enc_blk_nums'],
                        dec_blk_nums=opt['dec_blk_nums'])#.to(rank)
    elif name == 'NAFNet_Local':
        model = NAFNetLocal(img_channel=opt['img_channels'], 
                width=opt['width'], 
                middle_blk_num=opt['middle_blk_num'], 
                enc_blk_nums=opt['enc_blk_nums'],
                dec_blk_nums=opt['dec_blk_nums'])#.to(rank)

    else:
        raise NotImplementedError('This network is not implemented')
    if global_rank ==0:
        print( '**************************** \n',f'Using {name} network')

        input_size = (3, 1200, 1920)
        macs, params = get_model_complexity_info(model, input_size, print_per_layer_stat = False)
        print(f' ---- Computational complexity at {input_size}: {macs}')
        print(' ---- Number of parameters: ', params)    
    else:
        macs, params = None, None

    model.to(local_rank)

    #model = DDP(model, device_ids=[local_rank])
    # ========== 修改开始：安全地应用DDP包装 ==========
    import torch.distributed as dist
    # 判断是否真正需要并能够使用DDP
    # 条件：1. 分布式已初始化 2. 有多个GPU 3. 世界大小（进程数）>1
    use_ddp = dist.is_initialized() and torch.cuda.device_count() > 1 and int(os.environ.get('WORLD_SIZE', 1)) > 1
    
    if use_ddp:
        # 真正的多卡分布式训练模式
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if global_rank == 0:
            print(f"[INFO] 模型已包装为 DistributedDataParallel (DDP)，用于多GPU训练。")
    else:
        # 单卡模式：将模型移动到正确的设备（GPU或CPU）
        device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        if global_rank == 0:
            print(f"[INFO] 模型运行于单设备模式，设备: {device}。")
    # ========== 修改结束 ==========

    
    return model, macs, params
"""
def create_optim_scheduler(opt, model):
    '''
    Returns the optim and its scheduler.
    opt: a dictionary of the yaml config file with the train key
    '''
    optim = torch.optim.AdamW( filter(lambda p: p.requires_grad, model.parameters()) , 
                            lr = opt['lr_initial'],
                            weight_decay = opt['weight_decay'],
                            betas = opt['betas'])
    T_max = int(sum(opt['epochs']))

    if opt['lr_scheme'] == 'CosineAnnealing':
        scheduler = CosineAnnealingLR(optim, T_max=T_max, eta_min=opt['eta_min'])
    else: 
        raise NotImplementedError('scheduler not implemented')    
        
    return optim, scheduler

"""
def create_optim_scheduler(opt_train, model):
    """
    根据配置创建优化器和学习率调度器。
    Args:
        opt_train (dict): 训练配置字典。
        model (nn.Module): 需要优化的模型。
    Returns:
        optim, scheduler
    """
    # --- 1. 从配置中安全地获取参数，设置默认值 ---
    optim_config = opt_train.get('optim_g', {})
    lr = optim_config.get('lr', 2e-4)
    betas = optim_config.get('betas', (0.9, 0.999))
    weight_decay = optim_config.get('weight_decay', 0)

    # --- 2. 创建优化器 (使用Adam) ---
    # 注意：如果模型是DDP包装的，model.parameters() 会自动处理
    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    # --- 3. 创建学习率调度器 ---
    scheduler_config = opt_train.get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'CosineAnnealingRestartLR')

    if scheduler_type == 'CosineAnnealingRestartLR':
        # 余弦退火重启调度器 (常见于图像修复任务)
        periods = scheduler_config.get('periods', [250000])
        restart_weights = scheduler_config.get('restart_weights', [1])
        eta_min = scheduler_config.get('eta_min', 1e-7)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim,
            T_0=periods[0] if periods else 250000,
            T_mult=1,
            eta_min=eta_min,
            last_epoch=-1
        )
    elif scheduler_type == 'MultiStepLR':
        # 多步长下降调度器
        milestones = scheduler_config.get('milestones', [50000, 100000, 200000])
        gamma = scheduler_config.get('gamma', 0.5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=milestones, gamma=gamma)
    elif scheduler_type == 'StepLR':
        # 等步长下降调度器
        step_size = scheduler_config.get('step_size', 50000)
        gamma = scheduler_config.get('gamma', 0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=step_size, gamma=gamma)
    else:
        # 如果未指定或类型未知，使用最简化的LambdaLR（即恒定学习率）
        print(f"[INFO] 未识别的调度器类型 '{scheduler_type}'，将使用恒定学习率。")
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda epoch: 1.0)

    print(f"[INFO] 优化器已创建: Adam(lr={lr}) | 调度器: {scheduler_type}")
    return optim, scheduler


def load_weights(model, model_weights, global_rank = 1):
    '''
    Loads the weights of a pretrained model, picking only the weights that are
    in the new model.
    '''
    new_weights = model.state_dict()
    new_weights.update({k: v for k, v in model_weights.items() if k in new_weights})
    
    model.load_state_dict(new_weights)

    total_checkpoint_keys = len(model_weights)
    total_model_keys = len(new_weights)
    matching_keys = len(set(model_weights.keys()) & set(new_weights.keys()))

    if global_rank==0:
        print(f"Total keys in checkpoint: {total_checkpoint_keys}")
        print(f"Total keys in model state dict: {total_model_keys}")
        print(f"Number of matching keys: {matching_keys}")

    return model

def load_optim(optim, optim_weights):
    '''
    Loads the values of the optimizer picking only the weights that are in the new model.
    '''
    optim.load_state_dict(optim_weights)
    return optim

def resume_model(model,
                 optim,
                 scheduler, 
                 path_model, 
                 local_rank,
                 global_rank,resume:str=None):
    
    '''
    Returns the loaded weights of model and optimizer if resume flag is True
    '''
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    if resume:
        checkpoints = torch.load(path_model, map_location=map_location, weights_only=False)
        weights = checkpoints['model_state_dict']
        model = load_weights(model, model_weights=weights,global_rank = global_rank)

        start_epochs = 0
        if global_rank == 0: print(' ---- Loaded weights', '\n ***************************')
    else:
        start_epochs = 0
        if global_rank==0: print(' ---- Starting from zero the training', '\n ***************************')
    
    return model, optim, scheduler, start_epochs

def save_checkpoint(model, optim, scheduler, metrics_eval, metrics_train, paths, global_rank = 1):

    '''
    Save the .pt of the model after each epoch.
    
    '''
    if global_rank != 0: 
        return None
    
    if type(next(iter(metrics_eval.values()))) != dict:
        metrics_eval = {'metrics': metrics_eval}

    weights = model.state_dict()

    # Save the model after every epoch
    model_to_save = {
        'epoch': metrics_train['epoch'],
        'model_state_dict': weights,
        'optimizer_state_dict': optim.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }

    try:
        torch.save(model_to_save, paths)
        # print(f"Model saved to {paths['new']}")

    except Exception as e:
        print(f"Error saving model: {e}")


__all__ = ['create_model', 'resume_model', 'create_optim_scheduler', 'save_checkpoint',
           'load_optim', 'load_weights']



    
