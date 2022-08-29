from argparse import ArgumentParser
import yaml
import torch
from models.fourier1d import FNN1d_BSA_GradNorm
from train_utils import Adam
from train_utils.datasets import BSA_Loader_WithVirtualData, FES_Loader
from train_utils.train_2d import train_BSA_WithGradNorm
import os
from train_utils.losses_BSA import BSA_PINO_loss
import h5py
from train_utils.losses import LpLoss
from Defination_Experiments import Experiments_GradNorm_BSA, Experiments_Virtual_BSA
from scipy.io import savemat
# import spicy.io as io
import numpy as np

'''
The purpose of this runner is to train the corresponding PINO-MBD for the reliability assessment of the 4-storey 
building. There are three options for the 'Style' variable. Use Style=Train for training, Style=eval for generating a 
small amount of prediction results with a pretrained PINO-MBD, and Style=eval_batch to generate a large amount of 
prediction results for reliability assessment (STEP1).
After a large amount of predictions are made, turn to post [processing>Figure_RA.py] to continue STEP2 of reliability 
assessment.
Other details:
(1). The variable 'virtual_datapath' describes the location of the virtual data, which is generated in Matlab with
[Matlab codes>BSA>Generator_Virtual.m]
(2). The variable 'ckpt_path' describes the location of the pretrained PINO-MBD parameters.
(3). This example is trained on a small dataset. For best performance, use NoData: 'Off', Virtual Switch: 'On', 
and DiffLossSwitch: 'On'.
'''

f = open(r'configs/BSA/BSA_PINO-MBD.yaml')
BSA_config = yaml.load(f)

def run(config, args=False):
    data_config = config['data']
    ComDevice = torch.device('cuda:0')
    dataset = BSA_Loader_WithVirtualData(data_config['datapath'], data_config['weights_datapath'],
                                         data_config['test_datapath'], data_config['weights_datapath_test'],
                                         data_config['virtual_datapath'], data_config['weights_datapath_virtual'],
                                         data_config['Structure_datapath'],
                                         nt=data_config['nt'], nSlice=data_config['nSlice'],
                                         sub_t=data_config['sub_t'],
                                         new=False, inputDim=data_config['inputDim'],
                                         outputDim=data_config['outputDim'],
                                         ComDevice=ComDevice)

    train_loader, test_loader, virtual_loader, PDE_weights_virtual, ToOneV, W2_CX, W2_CY, W2_CZ, Eigens2, TrackDOFs, Nloc = dataset.make_loader(
        n_sample=data_config['n_sample'], n_sample_virtual=data_config['n_sample_virtual'],
        batch_size=config['train']['batchsize'],
        batch_size_virtual=config['train']['batchsize_virtual'],
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
    model = FNN1d_BSA_GradNorm(modes=config['model']['modes'],
                               width=config['model']['width'], fc_dim=config['model']['fc_dim'],
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    train_BSA_WithGradNorm(model,
                           train_loader, test_loader, virtual_loader, PDE_weights_virtual,
                           optimizer, scheduler,
                           config,
                           ToOneV,
                           W2_CX, W2_CY, W2_CZ,
                           Eigens2, TrackDOFs, Nloc,
                           inputDim=data_config['inputDim'], outputDim=data_config['outputDim'], D=data_config['D'],
                           ComDevice=ComDevice,
                           rank=0, log=False,
                           project='PINO-BSA',
                           group='default',
                           tags=['default'],
                           use_tqdm=True
                           )

    return model


Style = 'Train'
Multiple = 'Yes'
Clip = 5
File = './configs/BSA/BSA_PINO-MBD.yaml'
if Style == 'Train':
    Experiments_GradNorm_BSA(Multiple, Clip, File, run)
    # Experiments_Virtual_BSA(Multiple, Clip, File, run)
elif Style == 'eval':
    # Generate output as Matlab.mat file
    device = torch.device('cpu')
    BSA_data_config = BSA_config['data']
    model = FNN1d_BSA_GradNorm(modes=BSA_config['model']['modes'],
                               width=BSA_config['model']['width'], fc_dim=BSA_config['model']['fc_dim'],
                               inputDim=BSA_data_config['inputDim'],
                               outputDim=BSA_data_config['outputDim'], task_number=4).to(device)
    ckpt_path = '~.pt'
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'])
    virtual_datapath = '~.mat'
    input_eval = torch.tensor(h5py.File(virtual_datapath)['input'][:, 40:, :]).permute([2, 1, 0]).to(torch.float32)
    # eval_dataset = torch.utils.data.TensorDataset(input_eval)
    # eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1000, shuffle=False)
    index = 1
    SavePath = '~/'
    out = model(input_eval).detach().numpy()
    mdic = {"output": out}
    FileName = SavePath + 'Name.mat'
    savemat(FileName, mdic)

elif Style == 'eval_batch':
    # Generate a large amount of PINO-MBD prediction for reliability assessment
    device = torch.device('cpu')
    BSA_data_config = BSA_config['data']
    model = FNN1d_BSA_GradNorm(modes=BSA_config['model']['modes'],
                               width=BSA_config['model']['width'], fc_dim=BSA_config['model']['fc_dim'],
                               inputDim=BSA_data_config['inputDim'],
                               outputDim=BSA_data_config['outputDim'], task_number=4).to(device)
    # checkpoint file name
    ckpt_path = '~.pt'
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'])
    # input virtual dataset file name
    virtual_datapath = '~.mat'
    input_eval = torch.tensor(h5py.File(virtual_datapath)['input'][:, 40:, :]).permute([2, 1, 0]).to(torch.float32)
    eval_dataset = torch.utils.data.TensorDataset(input_eval)
    # choose the appropriate batch size according to your computer memory
    batch_size = 5000
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    eval_iter = iter(eval_loader)
    index = 1
    SavePath = '~/'
    # start iterative output prediction data
    for i in range(0, int(input_eval.size(0)/batch_size)+1):
        print('Now operating batch No.{}'.format(i+1))
        x = next(eval_iter)[0].to(device)
        out = model(x).detach()

        Name = SavePath + 'eval' + str(input_eval.size(0)) + '_' + str(i+1) + '.pt'
        torch.save(out, Name)

