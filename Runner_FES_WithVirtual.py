from argparse import ArgumentParser
import yaml
import torch
from models.fourier1d import FNN1d_FES
from train_utils import Adam
from train_utils.datasets import FES_Loader_WithVirtualData, FES_Loader
from train_utils.train_2d import train_FES_WithVirtual
from train_utils.eval_2d import eval_burgers
from train_utils.solution_extension import FDD_Extension
import matplotlib.pyplot as plt
import os
from train_utils.losses_FES import FES_PINO_loss
import numpy as np

'''
The purpose of this runner is to train the corresponding PINO-MBD for the simple examole in Extended Data Figure. 1. 
Since the main purpose of this runner is to evaluate the feasibility of non-data training. The following comments 
regarding config file should be emphasized.
(1). For non-data training with EN, use NoData: 'On', Virtual Switch: 'On', and DiffLossSwitch: 'Off'.
(2). For non-data training with EN and boundary restrains, use NoData: 'On', Virtual Switch: 'On', DiffLossSwitch: 'On',
 and Boundary: 'On'.
(3). For control group (trained with full data), use NoData: 'Off', Virtual Switch: 'On', DiffLossSwitch: 'On', and 
Boundary: 'Off'.
'''

f = open(r'configs/FES/FES.yaml')
FES_config = yaml.load(f)

def run(config, args=False):
    data_config = config['data']
    ComDevice = torch.device('cuda:0')
    dataset = FES_Loader_WithVirtualData(data_config['datapath'], data_config['weights_datapath'],
                                         data_config['test_datapath'], data_config['weights_datapath_test'],
                                         data_config['virtual_datapath'], data_config['weights_datapath_virtual'],
                                         data_config['Structure_datapath'],
                                         nt=data_config['nt'], nSlice=data_config['nSlice'],
                                         sub_t=data_config['sub_t'],
                                         new=False, inputDim=data_config['inputDim'],
                                         outputDim=data_config['outputDim'],
                                         ComDevice=ComDevice)

    # Manual:Change new to False(from new)
    train_loader, test_loader, virtual_loader, PDE_weights_virtual, ToOneV, W2, Eigens2, TrackDOFs, Nloc = dataset.make_loader(
        n_sample=data_config['n_sample'], n_sample_virtual=data_config['n_sample_virtual'],
        batch_size=config['train']['batchsize'],
        batch_size_virtual=config['train']['batchsize_virtual'],
        start=data_config['offset'])
    model = FNN1d_FES(modes=config['model']['modes'],
                      width=config['model']['width'], fc_dim=config['model']['fc_dim'],
                      inputDim=data_config['inputDim'],
                      outputDim=data_config['outputDim']).to(ComDevice)

    # Load from checkpoint
    if 'ckpt' in config['train']:
        ckpt_path = config['train']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)

    optimizer = Adam(model.parameters(), betas=(0.9, 0.999), lr=config['train']['base_lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.5)

    train_FES_WithVirtual(model,
                          train_loader, test_loader, virtual_loader, PDE_weights_virtual,
                          optimizer, scheduler,
                          config,
                          ToOneV,
                          W2, Eigens2, TrackDOFs, Nloc,
                          inputDim=data_config['inputDim'], outputDim=data_config['outputDim'], D=data_config['D'], ComDevice=ComDevice,
                          rank=0, log=False,
                          project='PINO-VTCD',
                          group='default',
                          tags=['default'],
                          use_tqdm=True
                          )

    return model


Style = 'Train'
if Style == 'Train':
    model = run(config=FES_config)
