from argparse import ArgumentParser
import yaml
import torch
from models.fourier1d import FNN1d_BSA
from train_utils import Adam
from train_utils.datasets import BSA_Loader_WithVirtualData, FES_Loader
from train_utils.train_2d import train_BSA_WithVirtual
from train_utils.eval_2d import eval_burgers
from train_utils.solution_extension import FDD_Extension
import matplotlib.pyplot as plt
import os
from train_utils.losses_BSA import BSA_PINO_loss

# import spicy.io as io
import numpy as np

f = open(r'configs/BSA/BSA.yaml')
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

    # Manual:Change new to False(from new)
    train_loader, test_loader, virtual_loader, PDE_weights_virtual, ToOneV, W2_CX, W2_CY, W2_CZ, Eigens2, TrackDOFs, Nloc = dataset.make_loader(
        n_sample=data_config['n_sample'], n_sample_virtual=data_config['n_sample_virtual'],
        batch_size=config['train']['batchsize'],
        batch_size_virtual=config['train']['batchsize_virtual'],
        start=data_config['offset'])
    model = FNN1d_BSA(modes=config['model']['modes'],
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
    # optimizer = torch.optim.SGD(model.parameters(), lr=config['train']['base_lr'], momentum=0.95, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                  milestones=config['train']['milestones'],
    #                                                  gamma=config['train']['scheduler_gamma'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1, eta_min=0.1 * config['train']['base_lr'])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.75)

    train_BSA_WithVirtual(model,
                          train_loader, test_loader, virtual_loader, PDE_weights_virtual,
                          optimizer, scheduler,
                          config,
                          ToOneV,
                          W2_CX, W2_CY, W2_CZ,
                          Eigens2, TrackDOFs, Nloc,
                          inputDim=data_config['inputDim'], outputDim=data_config['outputDim'], D=data_config['D'], ComDevice=ComDevice,
                          rank=0, log=False,
                          project='PINO-BSA',
                          group='default',
                          tags=['default'],
                          use_tqdm=True
                          )

    # for x, y in train_loader:
    #     x, y = x.cuda(), y.cuda()
    #     batch_size = config['train']['batchsize']
    #     nt = data_config['nt']
    #     out = model(x)
    #     print('Shape of x={}; Shape of y={}; Shape of out={}'.format(x.shape, y.shape, out.shape))
    #     device2 = torch.device('cpu')
    #     plt.figure(1)
    #     plt.plot(y.to(device2).permute([0, 2, 1])[0, 0, :].detach().numpy(), color='red')
    #     plt.plot(out.to(device2).permute([0, 2, 1])[0, 0, :].detach().numpy(), linestyle='--', color='black')
    #     plt.figure(2)
    #     plt.plot(y.to(device2).permute([0, 2, 1])[0, 0+20, :].detach().numpy(), color='red')
    #     plt.plot(out.to(device2).permute([0, 2, 1])[0, 0+20, :].detach().numpy(), linestyle='--', color='black')
    #     plt.figure(3)
    #     plt.plot(y.to(device2).permute([0, 2, 1])[0, 0+21, :].detach().numpy(), color='red')
    #     plt.plot(out.to(device2).permute([0, 2, 1])[0, 0+21, :].detach().numpy(), linestyle='--', color='black')
    #     plt.figure(4)
    #     plt.plot(y.to(device2).permute([0, 2, 1])[0, 0+22, :].detach().numpy(), color='red')
    #     plt.plot(out.to(device2).permute([0, 2, 1])[0, 0+22, :].detach().numpy(), linestyle='--', color='black')
    #     plt.figure(5)
    #     plt.plot(y.to(device2).permute([0, 2, 1])[0, 0+23, :].detach().numpy(), color='red')
    #     plt.plot(out.to(device2).permute([0, 2, 1])[0, 0+23, :].detach().numpy(), linestyle='--', color='black')
    #     plt.figure(6)
    #     plt.plot(y.to(device2).permute([0, 2, 1])[0, 0+26, :].detach().numpy(), color='red')
    #     plt.plot(out.to(device2).permute([0, 2, 1])[0, 0+26, :].detach().numpy(), linestyle='--', color='black')
    #     plt.show()
    #
    #     # dy, ddy = FDD_Extension(y, dt=0.5)
    #     # dout, ddout = FDD_Extension(out, dt=0.5)
    #     # # plt.figure(1)
    #     # # plt.plot(dy.to(device2).permute([0, 2, 1])[0, 0, :].detach().numpy(), color='red')
    #     # # plt.plot(dout.to(device2).permute([0, 2, 1])[0, 0, :].detach().numpy(), linestyle='--', color='black')
    #     # # plt.show()
    #     #
    #     # # dy = np.mat(dy.to(device2).detach().numpy())
    #     # # ddy = np.mat(ddy.to(device2).detach().numpy())
    #     # # dout = np.mat(dout.to(device2).detach().numpy())
    #     # # ddout = np.mat(ddout.to(device2).detach().numpy())
    #     # # io.savemat('PythonData.mat', {'dy': dy, 'ddy': ddy, 'dout': dout, 'ddout': ddout})
    #     #
    #     np.savetxt('y.txt', y.to(device2).permute([0, 2, 1])[0, 0+20, :].detach().numpy())
    #     np.savetxt('out.txt', out.to(device2).permute([0, 2, 1])[0, 0+20, :].detach().numpy())
    return model


def test(config, eval_model, args=False):
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    data_config = config['data']

    dataset = BSA_Loader(data_config['test_datapath'], data_config['testweights_datapath'],
                         data_config['test_datapath'], data_config['weights_datapath_test'],
                         data_config['Structure_datapath'],
                         nt=data_config['nt'], nSlice=data_config['nSlice'],
                         sub_t=data_config['sub_t'],
                         new=False, inputDim=data_config['inputDim'],
                         outputDim=data_config['outputDim'])

    # Manual:Change new to False(from new)
    _, test_loader, ToOneV, W2, Eigens2, TrackDOFs, Nloc = dataset.make_loader(
        n_sample=data_config['n_sample'],
        batch_size=config['train']['batchsize'],
        start=data_config['offset'])
    Index = 0
    # Define loss for all types of output
    Signal_Loss = 0.0
    First_Differential_Loss = 0.0
    Second_Differential_Loss = 0.0
    criterion = torch.nn.L1Loss(reduction='mean')
    for x, y, PDE_weights in test_loader:
        device2 = torch.device('cpu')
        x, y = x.to(device2), y.to(device2)
        DOF_exc = 3
        DOF_rigid = 9
        DOF_flex = int((1 / 3) * (y.shape[-1] - 3 * DOF_rigid))
        DOF_Serie = DOF_rigid + DOF_flex
        batch_size = config['train']['batchsize']
        inputDim = data_config['inputDim']
        outputDim = data_config['outputDim']
        nt = data_config['nt']
        out = model(x)
        loss_f, Du_Magnitude, Derivative_Data = BSA_PINO_loss(out, x, PDE_weights, ToOneV, inputDim, outputDim,
                                                              BSA_config['D'],
                                                              DOF_exc, DOF_rigid, DOF_flex)

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
        Signal_Loss += criterion(out, y[:, :, :DOF_Serie])
        First_Differential_Loss += criterion(Derivative_Data[:, :, :DOF_Serie], y[:, 1:-1, DOF_Serie:2 * DOF_Serie])
        Second_Differential_Loss += criterion(Derivative_Data[:, :, DOF_Serie:], y[:, 1:-1, 2 * DOF_Serie:])
        # plt.figure(1)
        # plt.plot(Derivative_Data[:, :, :outputDim][0, :, 0].detach().numpy())
        # plt.plot(y[:, 1:-1, outputDim:2*outputDim][0, :, 0].detach().numpy())
        # plt.show()
        # plt.figure(2)
        # plt.plot(Derivative_Data[:, :, outputDim:][0, :, 0].detach().numpy())
        # plt.plot(y[:, 1:-1, 2*outputDim:][0, :, 0].detach().numpy())
        # plt.show()
        np.savetxt(
            "checkpoints/" + config['train']['save_dir'] + "/Performance/GroundTruth_Batch" + str(Index) + ".txt",
            GroundTruth_Data)
        np.savetxt(
            "checkpoints/" + config['train']['save_dir'] + "/Performance/ModelOutput_Batch" + str(Index) + ".txt",
            ModelOutput_Data)
        Index += 1
    Signal_Loss /= len(test_loader)
    First_Differential_Loss /= len(test_loader)
    Second_Differential_Loss /= len(test_loader)
    print('Signal_Loss:{};First_Differential_Loss:{};Second_Differential_Loss:{}'.format(Signal_Loss,
                                                                                         First_Differential_Loss,
                                                                                         Second_Differential_Loss))


Style = 'Train'
if Style == 'Train':
    model = run(config=BSA_config)
else:
    device = torch.device('cpu')
    BSA_data_config = BSA_config['data']
    model = FNN1d_BSA(modes=BSA_config['model']['modes'],
                      width=BSA_config['model']['width'], fc_dim=BSA_config['model']['fc_dim'],
                      inputDim=BSA_data_config['inputDim'],
                      outputDim=BSA_data_config['outputDim']).to(device)
    if 'ckpt' in BSA_config['train']:
        ckpt_path = BSA_config['train']['ckpt']
        print(ckpt_path)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])

    test(config=BSA_config, eval_model=model)
