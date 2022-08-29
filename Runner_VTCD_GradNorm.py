from argparse import ArgumentParser
import yaml

import torch
from models.fourier1d import FNN1d_VTCD_GradNorm, FNN1d_VTCD
from train_utils import Adam
from train_utils.datasets import VTCD_Loader_WithVirtualData, VTCD_Loader_Variant1
from train_utils.train_2d import train_VTCD_GradNorm
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

'''
The purpose of this runner is to train the corresponding PINO-MBD for the vehicle-track coupled dynamics (VTCD).
Training details for the config file (in Table 1):
(1). For V1, use OperatorType: 'PINO-MBD', DiffLossSwitch: 'On', Boundary: 'Off', VirtualSwitch: 'Off'.
(2). For V2, use OperatorType: 'PINO-MBD', DiffLossSwitch: 'On', Boundary: 'On', VirtualSwitch: 'Off'.
(3). For V3, same as V2, use 'weights_datapath: 'Weights_PINO_10000V2.mat'' instead of 
'weights_datapath: 'data/Project_VTCD/Weights_10000V2.mat'.
(4). For V4, use use OperatorType: 'FNO'.
'''

f = open(r'configs/VTCD/VTCD_V2.yaml')
VTCD_config = yaml.load(f)

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
    model = FNN1d_VTCD_GradNorm(modes=config['model']['modes'],
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
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=0.1 * config['train']['base_lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.75)

    train_VTCD_GradNorm(model,
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


def test(config, eval_model, args=False):
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    data_config = config['data']
    dataset = VTCD_Loader_Variant1(data_config['test_datapath'], data_config['test_datapath'], nt=data_config['nt'],
                                   nSlice=data_config['nSlice'],
                                   sub_t=data_config['sub_t'],
                                   new=False, inputDim=data_config['inputDim'], outputDim=data_config['outputDim'])

    # Manual:Change new to False(from new)
    test_loader, test_loader_extra, ToOneV = dataset.make_loader(n_sample=data_config['test_sample'],
                                                                 batch_size=config['train']['batchsize'],
                                                                 start=data_config['offset'])
    Index = 0
    # Define loss for all types of output
    Signal_Loss = 0.0
    First_Differential_Loss = 0.0
    Second_Differential_Loss = 0.0
    criterion = torch.nn.L1Loss(reduction='mean')
    for x, y in test_loader:
        device2 = torch.device('cpu')
        x, y = x.to(device2), y.to(device2)
        batch_size = config['train']['batchsize']
        inputDim = VTCD_data_config['inputDim']
        outputDim = VTCD_data_config['outputDim']
        nt = data_config['nt']
        out = model(x)
        loss_f, Dummy, Derivative_Data = VTCD_PINO_loss_Variant1(out, x, ToOneV, inputDim, outputDim,
                                                                 VTCD_data_config['D'])
        # plt.figure('Carbody vertical displacement')
        # plt.plot(y.to(device2).permute([0, 2, 1])[0, 0, :].detach().numpy(), color='red')
        # plt.plot(out.to(device2).permute([0, 2, 1])[0, 0, :].detach().numpy(), linestyle='--', color='blue')
        # plt.figure('Carbody vertical velocity')
        # plt.plot(y.to(device2).permute([0, 2, 1])[0, 10, :].detach().numpy(), color='red')
        # plt.plot(out.to(device2).permute([0, 2, 1])[0, 10, :].detach().numpy(), linestyle='--', color='blue')
        # plt.figure('Wheelset vertical velocity')
        # plt.plot(y.to(device2).permute([0, 2, 1])[0, 16, :].detach().numpy(), color='red')
        # plt.plot(out.to(device2).permute([0, 2, 1])[0, 16, :].detach().numpy(), linestyle='--', color='blue')
        # plt.figure('Carbody vertical acceleration')
        # plt.plot(y.to(device2).permute([0, 2, 1])[0, 0 + 20, :].detach().numpy(), color='red')
        # plt.plot(out.to(device2).permute([0, 2, 1])[0, 0 + 20, :].detach().numpy(), linestyle='--', color='blue')
        # plt.figure('Carbody nod acceleration')
        # plt.plot(y.to(device2).permute([0, 2, 1])[0, 0 + 21, :].detach().numpy(), color='red')
        # plt.plot(out.to(device2).permute([0, 2, 1])[0, 0 + 21, :].detach().numpy(), linestyle='--', color='blue')
        # plt.figure('Bogie vertical acceleration')
        # plt.plot(y.to(device2).permute([0, 2, 1])[0, 0 + 22, :].detach().numpy(), color='red')
        # plt.plot(out.to(device2).permute([0, 2, 1])[0, 0 + 22, :].detach().numpy(), linestyle='--', color='blue')
        # plt.figure('Bogie nod acceleration')
        # plt.plot(y.to(device2).permute([0, 2, 1])[0, 0 + 23, :].detach().numpy(), color='red')
        # plt.plot(out.to(device2).permute([0, 2, 1])[0, 0 + 23, :].detach().numpy(), linestyle='--', color='blue')
        # plt.figure('Wheelset vertical acceleration')
        # plt.plot(y.to(device2).permute([0, 2, 1])[0, 0 + 26, :].detach().numpy(), color='red')
        # plt.plot(out.to(device2).permute([0, 2, 1])[0, 0 + 26, :].detach().numpy(), linestyle='--', color='blue')
        # plt.figure('Rail Displacement Under Wheelset')
        # plt.plot(y.to(device2).permute([0, 2, 1])[0, 0 + 29, :].detach().numpy(), color='red')
        # plt.plot(out.to(device2).permute([0, 2, 1])[0, 0 + 29, :].detach().numpy(), linestyle='--', color='blue')
        # plt.figure('Rail Displacement Under Wheelset')
        # plt.plot(y.to(device2).permute([0, 2, 1])[0, 0 + 30, :].detach().numpy(), color='red')
        # plt.plot(out.to(device2).permute([0, 2, 1])[0, 0 + 30, :].detach().numpy(), linestyle='--', color='blue')
        # plt.show()
        GroundTruth_Data = y.to(device2).permute([0, 2, 1])[0, :, :].detach().numpy()
        ModelOutput_Data = out.to(device2).permute([0, 2, 1])[0, :, :].detach().numpy()
        Signal_Loss += criterion(out, torch.cat([y[:, :, :10], y[:, :, -4:]], dim=-1))
        First_Differential_Loss += criterion(Derivative_Data[:, :, :10], y[:, 1:-1, 10:20])
        Second_Differential_Loss += criterion(Derivative_Data[:, :, 10:], y[:, 1:-1, 20:30])
        # plt.figure(1)
        # plt.plot(Derivative_Data[:, :, :outputDim][0, :, 0].detach().numpy())
        # plt.plot(y[:, 1:-1, outputDim:2*outputDim][0, :, 0].detach().numpy())
        # plt.show()
        # plt.figure(2)
        # plt.plot(Derivative_Data[:, :, outputDim:][0, :, 0].detach().numpy())
        # plt.plot(y[:, 1:-1, 2*outputDim:][0, :, 0].detach().numpy())
        # plt.show()
        # np.savetxt("visualization/Analysis_DataSet_Style/MCM/GroundTruth_Batch" + str(Index) + ".txt", GroundTruth_Data)
        # np.savetxt("visualization/Analysis_DataSet_Style/MCM/ModelOutput_Batch" + str(Index) + ".txt", ModelOutput_Data)
        Index += 1
    Signal_Loss /= len(test_loader)
    First_Differential_Loss /= len(test_loader)
    Second_Differential_Loss /= len(test_loader)
    print('Signal_Loss:{};First_Differential_Loss:{};Second_Differential_Loss:{}'.format(Signal_Loss,
                                                                                         First_Differential_Loss,
                                                                                         Second_Differential_Loss))


Style = 'eval'
Multiple = 'Yes'
Clip = 5
File = './configs/VTCD/VTCD_V2.yaml'
if Style == 'Train':
    time.sleep(1)
    Experiments_GradNorm_VTCD(Multiple, Clip, File, run)
else:
    device = torch.device('cpu')
    VTCD_data_config = VTCD_config['data']
    model = FNN1d_VTCD_GradNorm(modes=VTCD_config['model']['modes'],
                       width=VTCD_config['model']['width'], fc_dim=VTCD_config['model']['fc_dim'],
                       inputDim=VTCD_data_config['inputDim'],
                       outputDim=VTCD_data_config['outputDim'], task_number=1).to(device)

    if 'ckpt' in VTCD_config['train']:
        ckpt_path = VTCD_config['train']['ckpt']
        ckpt = torch.load(ckpt_path)
        # o = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/VTCDRunner/VTCD_FNO.pt'
        # print(torch.load(o))
        model.load_state_dict(ckpt['model'])
    data_config = VTCD_config['data']
    dataset = VTCD_Loader_WithVirtualData(data_config['datapath'], data_config['weights_datapath'],
                                          data_config['test_datapath'],
                                          data_config['virtual_datapath'], data_config['weights_datapath_virtual'],
                                          nt=data_config['nt'], nSlice=data_config['nSlice'],
                                          sub_t=data_config['sub_t'],
                                          new=False, inputDim=data_config['inputDim'],
                                          outputDim=data_config['outputDim'])
    train_loader, test_loader, virtual_loader, PDE_weights_virtual, ToOneV = dataset.make_loader(
        n_sample=data_config['n_sample'], n_sample_virtual=data_config['n_sample_virtual'],
        batch_size=VTCD_config['train']['batchsize'], batch_size_virtual=VTCD_config['train']['batchsize_virtual'],
        start=data_config['offset'])
    Switch1 = 'On'
    if Switch1 == 'On':
        eval_iter = iter(test_loader)
        x, y = next(eval_iter)
        out = model(x).detach()
        SavePath = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/VTCDRunner/'
        Name = 'PhysicsUninformed_Performance.txt'
        Name = SavePath + Name
        output = torch.cat([out[0, :, :], torch.cat([y[0, :, :10], y[0, :, -4:]], dim=-1)], dim=-1)
        print(output.shape)
        np.savetxt(Name, output.numpy())

    Switch2 = 'Off'
    if Switch2 == 'On':
        # test(config=VTCD_config, eval_model=model)
        # Scale the losses for all different components
        Scale_loss = np.zeros((10, 3))
        batch_number = 0
        myloss = LpLoss(size_average=True)
        for x, y in test_loader:
            print('Operating batch No.{}'.format(batch_number))
            out = model(x)
            test_weights = PDE_weights_virtual.double().repeat_interleave(x.size(0), dim=0)
            _, _, Derivative_Data = VTCD_PINO_loss(out, x, test_weights, ToOneV, VTCD_config['data']['inputDim'],
                                                   VTCD_config['data']['outputDim'], VTCD_config['data']['D'], device)
            De0_GT = y[:, :, :10]
            De1_GT = y[:, 1:-1, 10:20]
            De2_GT = y[:, 1:-1, 20:-4]
            De0_Pre = out[:, :, :10]
            De1_Pre = Derivative_Data[:, :, :10]
            De2_Pre = Derivative_Data[:, :, 10:]

            for i in range(0, 10):
                for j in range(0, 3):
                    if j == 0:
                        Fruit1 = De0_GT[:, :, i]
                        Fruit2 = De0_Pre[:, :, i]
                    elif j == 1:
                        Fruit1 = De1_GT[:, :, i]
                        Fruit2 = De1_Pre[:, :, i]
                    elif j == 2:
                        Fruit1 = De2_GT[:, :, i]
                        Fruit2 = De2_Pre[:, :, i]
                    Scale_loss[i, j] = Scale_loss[i, j] + myloss(Fruit1, Fruit2).detach().numpy()

            batch_number += 1
        Scale_loss = Scale_loss / batch_number
        print(Scale_loss)
        SavePath = 'E:/学习库/文章/PINO-MBD_Physics-Informed Neual Operator for multi-body dynamics/Nature文献/Draft/Figures/Fig3/'
        Name = 'WithoutEN_Scale.txt'
        Name = SavePath + Name
        np.savetxt(Name, Scale_loss)
