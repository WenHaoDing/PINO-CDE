from argparse import ArgumentParser

import pandas as pd
import yaml
import torch
import shutil
from models.fourier1d import FNN1d_HM
from train_utils import Adam
from train_utils.datasets import HM_Loader_WithVirtualData
from train_utils.train_2d import train_HM_WithVirtual, train_HM_LRA
from train_utils.eval_2d import eval_burgers
from train_utils.solution_extension import FDD_Extension
import matplotlib.pyplot as plt
import os
from train_utils.losses_HM import HM_PINO_loss
from Defination_Experiments import Experiments_GradNorm, Experiments_Virtual
from Defination_Experiments_V2 import GainAnalysis_HM
# import spicy.io as io
import numpy as np
from HPA_tool.Hyperparameter_Analysis import read_HPA_table, yaml_update_HM

"""
explore the learning rate annealing algorithm proposed by Sifan Wang
"""


def run(config, args=False):
    data_config = config['data']
    ComDevice = torch.device('cuda:0')
    dataset = HM_Loader_WithVirtualData(data_config['datapath'], data_config['weights_datapath'],
                                        data_config['test_datapath'], data_config['weights_datapath_test'],
                                        data_config['virtual_datapath'], data_config['weights_datapath_virtual'],
                                        data_config['Structure_datapath'],
                                        nt=data_config['nt'], nSlice=data_config['nSlice'],
                                        sub_t=data_config['sub_t'],
                                        new=False, inputDim=data_config['inputDim'],
                                        outputDim=data_config['outputDim'],
                                        ComDevice=ComDevice)

    # Manual:Change new to False(from new)
    train_loader, test_loader, virtual_loader, PDE_weights_virtual, ToOneV, W2_1, W2_2, W2_3, Eigens2_1, Eigens2_2 = dataset.make_loader(
        n_sample=data_config['n_sample'], n_sample_virtual=data_config['n_sample_virtual'],
        batch_size=config['train']['batchsize'],
        batch_size_virtual=config['train']['batchsize_virtual'])
    virtual_iter = iter(virtual_loader)
    if data_config['OperatorType'] == 'PINO-MBD' or data_config['OperatorType'] == 'PINO':
        if data_config['NoData'] == 'On':
            task_number = 1
        else:
            task_number = 2
            if data_config['DiffLossSwitch'] == 'On':
                task_number += 1
            if data_config['VirtualSwitch'] == 'On':
                task_number += 1
    else:
        task_number = 1

    print('This mission will have {} task(s)'.format(task_number))
    if data_config['GradNorm'] == 'On' and task_number != 1:
        print('GradNorm will be launched with alpha={}.'.format(data_config['GradNorm_alpha']))
    else:
        print('GradNorm will not be launched for this mission.')
    print(config['model']['modes'])
    print(config['model']['depth'])
    model = FNN1d_HM(modes=config['model']['modes'], depth=config['model']['depth'],
                     width=config['model']['width'], fc_dim=config['model']['fc_dim'], fc_dep=config['model']['fc_dep'],
                     inputDim=data_config['inputDim'],
                     outputDim=data_config['outputDim'],
                     task_number=task_number).to(ComDevice)

    # Load from checkpoint
    if 'ckpt' in config['train']:
        ckpt_path = config['train']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)

    optimizer = Adam(model.parameters(), betas=(0.9, 0.999), lr=config['train']['base_lr'])
    # optimizer = torch.optim.SGD(model.parameters(), lr=config['train']['base_lr'], momentum=0.95, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                  milestones=config['train']['milestones'],
    #                                                  gamma=config['train']['scheduler_gamma'])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1, eta_min=0.1 * config['train']['base_lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.5)

    train_HM_LRA(model,
                 train_loader, test_loader, virtual_loader, PDE_weights_virtual,
                 optimizer, scheduler,
                 config,
                 ToOneV,
                 W2_1, W2_2, W2_3, Eigens2_1, Eigens2_2,
                 inputDim=data_config['inputDim'], outputDim=data_config['outputDim'], D=data_config['D'],
                 ComDevice=ComDevice,
                 rank=0, log=False,
                 project='PINO-VTCD',
                 group='default',
                 tags=['default'],
                 use_tqdm=True
                 )
    return model


Style = 'Train'
Multiple = 'No'
Clip = 3
File = './configs/HM/HM_PINO-MBD.yaml'
if Style == 'Train':
    File = './configs/HM/HM_LRA.yaml'
    Table = './checkpoints/HMRunner/LRA/HM_LRA.xls'
    HPA_table = read_HPA_table(Table)
    Yaml_File = './configs/HM/HM_LRA.yaml'
    Case_start = 1
    Case_End = 6
    CasePack = []
    for i in range(Case_start, Case_End + 1):
        CasePack += [i]

    for case in CasePack:
        yaml_update_HM(HPA_table[case - 1], Yaml_File)
        f = open(r'configs/HM/HM_LRA.yaml')
        HM_config = yaml.safe_load(f)
        _ = run(config=HM_config)
