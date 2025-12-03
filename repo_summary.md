# Repository Structure Summary

This document provides a summary of the directory and file structure of this repository.

## Project Overview

This repository is the official codebase for the **[AIM 2025] Efficient Real-World Deblurring Challenge**. It provides a baseline framework for training deblurring models and a pipeline for calculating the computational cost of the models.

The main functionalities are divided into three tasks:
- **Training**: `train.sh` and `train.py`
- **Inference**: `inference.sh` and `inference.py`
- **Computing Cost**: `cost.sh` and `computing_cost.py`

The project is tested with Python 3.10.12, PyTorch >= 2.3.1, and CUDA 12.4. The dataset used is [RSBlur](https://github.com/rimchang/RSBlur). The baseline implementation is based on [NAFNet](https://github.com/megvii-research/NAFNet).

## Directory Structure

* **archs/**: Contains the different architectures for the models.
* **data/**: Contains datasets and data loading scripts.
* **losses/**: Contains different loss functions.
* **models/**: Contains the model definitions.
* **options/**: Contains configuration options for training and testing.
* **results/**: Directory to save the results of experiments.
* **tools/**: Contains utility scripts for various tasks.
* **utils/**: Contains utility functions used across the repository.

## File Structure

* **.gitignore**: Specifies which files and directories to ignore in Git.
* **README.md**: Provides an overview of the project.
* **computing_cost.py**: Script for computing the cost of a model.
* **cost.sh**: Shell script related to computing cost.
* **create_all_index_files.py**: Script to create all index files.
* **create_index_files.py**: Script to create index files.
* **create_paired_dataset.py**: Script to create a paired dataset.
* **inference.py**: Script for running inference.
* **inference.sh**: Shell script for running inference.
* **requirements.txt**: Lists the Python dependencies for the project.
* **testingDatePath.py**: Script for testing date paths.
* **train.py**: The main script for training the model.
* **train.sh**: Shell script for training the model.
