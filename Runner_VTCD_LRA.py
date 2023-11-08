from argparse import ArgumentParser
import yaml

import torch
from models.fourier1d import FNN1d_VTCD_LRA, FNN1d_VTCD
from train_utils import Adam
from train_utils.datasets import VTCD_Loader_WithVirtualData, VTCD_Loader_Variant1
from train_utils.train_2d import train_VTCD_LRA
from train_utils.eval_2d import eval_burgers
from train_utils.solution_extension import FDD_Extension
import matplotlib.pyplot as plt
import os
from train_utils.losses_VTCD import VTCD_PINO_loss, VTCD_PINO_loss_Variant1
from train_utils.losses import LpLoss
# import spicy.io as io
import numpy as np
import time
from Defination_Experiments import Experiments_GradNorm_VTCD
from HPA_tool.Hyperparameter_Analysis import read_HPA_table, yaml_update_VTCD
from train_utils.losses import LpLoss


# f = open(r'configs\pretrain\VTCD.yaml')
os.environ["CUDA_VISIBLE_DEVICES"] = str(input('select GPU ID= '))

def run(config, args=False):
    data_config = config['data']
    ComDevice = torch.device('cuda:0')
    dataset = VTCD_Loader_WithVirtualData(data_config['datapath'], data_config['weights_datapath'],
                                          data_config['test_datapath'],
                                          data_config['virtual_datapath'], data_config['weights_datapath_virtual'],
                                          nt=data_config['nt'], nSlice=data_config['nSlice'],
                                          sub_t=data_config['sub_t'],
                                          new=False, inputDim=data_config['inputDim'],
                                          outputDim=data_config['outputDim'])

    # Manual:Change new to False(from new)
    train_loader, test_loader, virtual_loader, PDE_weights_Virtual, ToOneV = dataset.make_loader(
        n_sample=data_config['n_sample'], n_sample_virtual=data_config['n_sample_virtual'],
        batch_size=config['train']['batchsize'], batch_size_virtual=config['train']['batchsize_virtual'],
        start=data_config['offset'])
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
    model = FNN1d_VTCD_LRA(modes=config['model']['modes'], depth=config['model']['depth'],
                           width=config['model']['width'], fc_dim=config['model']['fc_dim'],
                           fc_dep=config['model']['fc_dep'],
                           inputDim=data_config['inputDim'],
                           outputDim=data_config['outputDim']).to(ComDevice)

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
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=0.1 * config['train']['base_lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.75)

    train_VTCD_LRA(model,
                   train_loader, test_loader, virtual_loader, PDE_weights_Virtual,
                   optimizer, scheduler,
                   config,
                   ToOneV=ToOneV,
                   inputDim=data_config['inputDim'], outputDim=data_config['outputDim'],
                   D=data_config['D'], ComDevice=ComDevice,
                   rank=0, log=False,
                   project='PINO-VTCD',
                   group='default',
                   tags=['default'],
                   use_tqdm=True)
    return model


Style = 'train'

if Style == 'train':
    File = './configs/VTCD/VTCD_LRA.yaml'
    Table = './checkpoints/VTCDRunner/LRA/VTCD_LRA.xls'
    HPA_table = read_HPA_table(Table)
    Yaml_File = File
    Case_start = 3
    Case_End = 3
    CasePack = []
    for i in range(Case_start, Case_End + 1):
        CasePack += [i]
    CasePack = [3, 6]
    for case in CasePack:
        yaml_update_VTCD(HPA_table[case - 1], Yaml_File)
        f = open(r'configs/VTCD/VTCD_LRA.yaml')
        config = yaml.safe_load(f)
        if config['data']['DiffLossSwitch'] == 'On':
            task_number = 3
        else:
            task_number = 2

        _ = run(config)
