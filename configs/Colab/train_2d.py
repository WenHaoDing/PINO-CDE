import numpy as np
import torch
from tqdm import tqdm
from .utils import save_checkpoint
from .losses import LpLoss, darcy_loss, PINO_loss
from .losses_FES import FES_PINO_loss
from .losses_AMA import AMA_PINO_loss
from .losses_VTCD import VTCD_PINO_loss, VTCD_PINO_loss_Variant1
from .losses_BSA import BSA_PINO_loss
import os

try:
    import wandb
except ImportError:
    wandb = None


def train_FES(model,
              train_loader, test_loader,
              optimizer, scheduler,
              config,
              ToOneV,
              W2, Eigens2, TrackDOFs, Nloc,
              inputDim, outputDim, D,
              rank=0, log=False,
              project='PINO-VTCD',
              group='default',
              tags=['default'],
              use_tqdm=False
              ):
    """
    Felexible model Example No.1
    Mapping only the solution (displacement)
    Offering the Ground Truth of the first and second derivative
    Using PDE restrain (derivative coming from the solution)
    """
    if rank == 0 and wandb and log:
        run = wandb.init(project=project,
                         entity='hzzheng-pino',
                         group=group,
                         config=config,
                         tags=tags, reinit=True,
                         settings=wandb.Settings(start_method="fork"))

    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    ic_weight = config['train']['ic_loss']
    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    # Record of loss functions
    LossRecord = np.zeros((config['train']['epochs'] + 1, 6 + 24 * 2))
    Index = 0

    for e in pbar:
        model.train()
        train_pino = 0.0
        data_l2 = 0.0
        train_loss = 0.0
        Magnitude = 0.0
        DiffRelation_loss = 0.0

        train_pino_test = 0.0
        data_l2_test = 0.0
        train_loss_test = 0.0
        Magnitude_test = np.zeros((1, 2 * (15 + 9)))
        DiffRelation_loss_test = 0.0
        SignalLoss_test = 0.0
        FirstDerivativeLoss_test = 0.0
        SecondDerivativeLoss_test = 0.0

        for x, y, PDE_weights in train_loader:
            x, y = x.to(rank), y.to(rank)
            DOF_exc = 3
            DOF_rigid = 9
            DOF_flex = int((1 / 3) * (y.shape[-1] - 3 * DOF_rigid))
            DOF_Serie = DOF_rigid + DOF_flex

            # ManualSpy
            # print('Shape of input is{}; output is{} and model output is{}'.format(x.shape, y.shape, model(x).shape))
            out = model(x).reshape(y.size(0), y.size(1), outputDim)
            data_loss = myloss(out[:, :50, :], y[:, :50, :DOF_Serie])  # Pure data losses-Only with first data
            # Direct PDE loss of the signal
            loss_f, Du_Magnitude, Derivative_Data = FES_PINO_loss(out, x, PDE_weights, ToOneV, inputDim, outputDim, D,
                                                                  DOF_exc, DOF_rigid, DOF_flex, W2, Eigens2, TrackDOFs,
                                                                  Nloc)
            # loss_f = torch.tensor([0]).to(torch.device('cuda:0'))
            # Du_Magnitude = torch.tensor([0])
            # Diff Relation loss for the signal pack
            criterion = torch.nn.L1Loss()
            # loss_DiffRelation = criterion(Derivative_Data, y[:, 1:-1, DOF_Serie:])
            # loss_DiffRelation = myloss(Derivative_Data[:, :, :DOF_Serie], y[:, 1:-1, DOF_Serie: 2*DOF_Serie]) + myloss(Derivative_Data[:, :, DOF_Serie:], y[:, 1:-1, 2*DOF_Serie:])
            loss_DiffRelation = myloss(Derivative_Data[:, :50, :], y[:, 1:51, DOF_Serie:])

            # loss_DiffRelation = torch.tensor([0]).to(torch.device('cuda:0'))

            # total_loss = loss_f * f_weight + data_loss * data_weight + 1.0 * loss_DiffRelation
            total_loss = data_loss * data_weight + 1.0 * loss_DiffRelation

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            data_l2 += data_loss.item()
            train_pino += loss_f.item()
            train_loss += total_loss.item()
            DiffRelation_loss += loss_DiffRelation.item()
            Magnitude_test += Du_Magnitude.detach().to(torch.device('cpu')).numpy()
            # Magnitude = (torch.max(Du_Magnitude) / torch.min(Du_Magnitude)).item()

        for x, y, PDE_weights in test_loader:
            x, y = x.to(rank), y.to(rank)
            # ManualSpy
            out = model(x).reshape(y.size(0), y.size(1), outputDim)
            data_loss = myloss(out, y[:, :, :DOF_Serie]).detach().to(torch.device('cpu')).numpy()
            # Direct PDE loss of the signal
            loss_f, Du_Magnitude, Derivative_Data = FES_PINO_loss(out, x, PDE_weights, ToOneV, inputDim, outputDim, D,
                                                                  DOF_exc, DOF_rigid, DOF_flex, W2, Eigens2, TrackDOFs,
                                                                  Nloc)
            # loss_f = torch.tensor([0]).to(torch.device('cuda:0'))
            # Du_Magnitude = torch.tensor([0])
            # Diff Relation loss for the signal pack
            # Diff_Weight = torch.ones(20).to(torch.device('cuda:0'))
            # Diff_Weight[0], Diff_Weight[10] = 2, 2
            # loss_DiffRelation = myloss(Derivative_Data, y[:, 1:-1, DOF_Serie:])
            loss_DiffRelation = torch.tensor([0]).to(torch.device('cuda:0'))

            FirstDerivative_loss = myloss(Derivative_Data[:, :, :DOF_Serie],
                                          y[:, 1:-1, DOF_Serie:2 * DOF_Serie]).detach().to(
                torch.device('cpu')).numpy()
            SecondDerivative_loss = myloss(Derivative_Data[:, :, DOF_Serie:], y[:, 1:-1, 2 * DOF_Serie:]).detach().to(
                torch.device('cpu')).numpy()
            total_loss = data_loss + FirstDerivative_loss + SecondDerivative_loss

            # total_loss = data_loss * data_weight
            train_pino_test += loss_f.item()
            train_loss_test += total_loss.item()
            DiffRelation_loss_test += loss_DiffRelation.item()
            SignalLoss_test += data_loss
            FirstDerivativeLoss_test += FirstDerivative_loss
            SecondDerivativeLoss_test += SecondDerivative_loss

        scheduler.step()
        data_l2 /= len(train_loader)
        train_pino /= len(train_loader)
        train_loss /= len(train_loader)
        DiffRelation_loss /= len(train_loader)
        Magnitude_test /= len(train_loader)

        train_pino_test /= len(test_loader)
        train_loss_test /= len(test_loader)
        DiffRelation_loss_test /= len(test_loader)
        SignalLoss_test /= len(test_loader)
        FirstDerivativeLoss_test /= len(test_loader)
        SecondDerivativeLoss_test /= len(test_loader)
        # if use_tqdm:
        #     pbar.set_description(
        #         (
        #             f'Epoch {e}, train loss: {train_loss:.5f} '
        #             f'train f error: {train_pino * f_weight:.5f}; '
        #             f'data l2 error: {data_l2:.5f}; '
        #             f'Magnitude Situation: {Magnitude:.1f}'
        #         )
        #     )
        # if wandb and log:
        #     wandb.log(
        #         {
        #             'Train f error': train_pino,
        #             'Train L2 error': data_l2,
        #             'Train loss': train_loss,
        #         }
        #     )
        LossRecord[Index, 0:6] = [DiffRelation_loss, train_pino * f_weight, data_l2,
                                  SignalLoss_test, FirstDerivativeLoss_test, SecondDerivativeLoss_test]
        LossRecord[Index, 6:] = Magnitude_test
        Index += 1
        if e % 250 == 0:
            save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                            model, optimizer)
        if e % 2 == 0:
            np.savetxt('checkpoints/' + config['train']['save_dir'] + '/' + 'Performance' + str(Index) + '.txt',
                       torch.cat([out[0, :, :], y[0, :, :DOF_Serie]], dim=-1).detach().to(
                           torch.device('cpu')).numpy())
            np.savetxt('checkpoints/' + config['train']['save_dir'] + '/LossHistory.txt', LossRecord)
        if Index >= 50:
            if np.sum(LossRecord[Index, -3:]) < np.sum(LossRecord[Index - 1, -3:]):
                save_checkpoint(config['train']['save_dir'],
                                config['train']['save_name'].replace('.pt', 'Best_On_Test.pt'),
                                model, optimizer)
        if Index >= 275:
            if e % 2 == 0:
                save_checkpoint(config['train']['save_dir'],
                                config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                                model, optimizer)
        print('train loss:{}; f loss:{}; DiffRelation loss:{}, data loss:{}'.format(train_loss,
                                                                                    train_pino * f_weight,
                                                                                    1.0 * DiffRelation_loss,
                                                                                    data_l2))
        print('test loss:{}; Signal loss:{}; FirstDerivative loss:{}, SecondDerivative loss:{}'.format(train_loss_test,
                                                                                                       SignalLoss_test,
                                                                                                       FirstDerivativeLoss_test,
                                                                                                       SecondDerivativeLoss_test))
        # print('Magnitude of PDE loss for different equations:', Magnitude_test)
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    print('Done!')


def train_FES_WithVirtual(model,
                          train_loader, test_loader, virtual_loader, PDE_weights_virtual,
                          optimizer, scheduler,
                          config,
                          ToOneV,
                          W2, Eigens2, TrackDOFs, Nloc,
                          inputDim, outputDim, D, ComDevice,
                          rank=0, log=False,
                          project='PINO-VTCD',
                          group='default',
                          tags=['default'],
                          use_tqdm=False
                          ):
    """
    Felexible model Example No.1
    Mapping only the solution (displacement)
    Offering the Ground Truth of the first and second derivative
    Using PDE restrain (derivative coming from the solution)
    """
    if rank == 0 and wandb and log:
        run = wandb.init(project=project,
                         entity='hzzheng-pino',
                         group=group,
                         config=config,
                         tags=tags, reinit=True,
                         settings=wandb.Settings(start_method="fork"))

    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    fv_weight = config['train']['fv_loss']
    diff_weight = config['train']['diff_loss']
    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    # Record of loss functions
    LossRecord = np.zeros((config['train']['epochs'] + 1, 7 + 24 * 2))
    Index = 0

    for e in pbar:
        model.train()
        train_pino = 0.0
        virtual_pino = 0.0
        data_l2 = 0.0
        train_loss = 0.0
        Magnitude = 0.0
        DiffRelation_loss = 0.0

        train_pino_test = 0.0
        data_l2_test = 0.0
        train_loss_test = 0.0
        Magnitude_test = np.zeros((1, 2 * (15 + 9)))
        DiffRelation_loss_test = 0.0
        SignalLoss_test = 0.0
        FirstDerivativeLoss_test = 0.0
        SecondDerivativeLoss_test = 0.0
        VirtualSwitch = config['data']['VirtualSwitch']
        PerformanceSwitch = config['data']['PerformanceSwitch']

        for x, y, PDE_weights in train_loader:
            x, y = x.to(ComDevice), y.to(ComDevice)
            DOF_exc = 3
            DOF_rigid = 9
            DOF_flex = int((1 / 3) * (y.shape[-1] - 3 * DOF_rigid))
            DOF_Serie = DOF_rigid + DOF_flex

            # ManualSpy
            # print('Shape of input is{}; output is{} and model output is{}'.format(x.shape, y.shape, model(x).shape))
            out = model(x).reshape(y.size(0), y.size(1), outputDim)
            data_loss = myloss(out[:, :50, :], y[:, :50, :DOF_Serie])  # Pure data losses
            # Direct PDE loss of the signal
            loss_f, Du_Magnitude, Derivative_Data = FES_PINO_loss(out, x, PDE_weights, ToOneV, inputDim, outputDim, D,
                                                                  DOF_exc, DOF_rigid, DOF_flex, W2, Eigens2, TrackDOFs,
                                                                  Nloc, ComDevice)
            if VirtualSwitch == 'On':
                x_virtual = next(iter(virtual_loader))[0].to(ComDevice)
                out_virtual = model(x_virtual).reshape(x_virtual.size(0), y.size(1), outputDim)
                virtual_weights = PDE_weights_virtual.double().repeat_interleave(x_virtual.size(0), dim=0)
                loss_f_virtual, _, _ = FES_PINO_loss(out_virtual, x_virtual, virtual_weights, ToOneV, inputDim,
                                                     outputDim,
                                                     D, DOF_exc, DOF_rigid, DOF_flex, W2, Eigens2, TrackDOFs, Nloc,
                                                     ComDevice)
            else:
                loss_f_virtual = torch.tensor([0]).to(ComDevice)
            # loss_f = torch.tensor([0]).to(torch.device('cuda:0'))
            # Du_Magnitude = torch.tensor([0])
            # Diff Relation loss for the signal pack
            criterion = torch.nn.L1Loss()
            # loss_DiffRelation = criterion(Derivative_Data, y[:, 1:-1, DOF_Serie:])
            # loss_DiffRelation = myloss(Derivative_Data[:, :, :DOF_Serie], y[:, 1:-1, DOF_Serie: 2*DOF_Serie]) + myloss(Derivative_Data[:, :, DOF_Serie:], y[:, 1:-1, 2*DOF_Serie:])
            loss_DiffRelation = myloss(Derivative_Data[:, :50, :], y[:, 1:51, DOF_Serie:])

            # loss_DiffRelation = torch.tensor([0]).to(torch.device('cuda:0'))

            total_loss = loss_f * f_weight + loss_f_virtual * fv_weight + data_loss * data_weight + loss_DiffRelation * diff_weight
            # total_loss = loss_f_virtual * fv_weight + data_loss * data_weight + loss_DiffRelation * diff_weight
            # total_loss = data_loss * data_weight + 1.0 * loss_DiffRelation

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            data_l2 += data_loss.item()
            train_pino += loss_f.item()
            virtual_pino += loss_f_virtual.item()
            train_loss += total_loss.item()
            DiffRelation_loss += loss_DiffRelation.item()
            Magnitude_test += Du_Magnitude.detach().to(torch.device('cpu')).numpy()
            # Magnitude = (torch.max(Du_Magnitude) / torch.min(Du_Magnitude)).item()

        for x, y, PDE_weights in test_loader:
            x, y = x.to(ComDevice), y.to(ComDevice)
            # ManualSpy
            out = model(x).reshape(y.size(0), y.size(1), outputDim)
            data_loss = myloss(out, y[:, :, :DOF_Serie]).detach().to(torch.device('cpu')).numpy()
            # Direct PDE loss of the signal
            loss_f, Du_Magnitude, Derivative_Data = FES_PINO_loss(out, x, PDE_weights, ToOneV, inputDim, outputDim, D,
                                                                  DOF_exc, DOF_rigid, DOF_flex, W2, Eigens2, TrackDOFs,
                                                                  Nloc, ComDevice)
            # loss_f = torch.tensor([0]).to(torch.device('cuda:0'))
            # Du_Magnitude = torch.tensor([0])
            # Diff Relation loss for the signal pack
            # Diff_Weight = torch.ones(20).to(torch.device('cuda:0'))
            # Diff_Weight[0], Diff_Weight[10] = 2, 2
            # loss_DiffRelation = myloss(Derivative_Data, y[:, 1:-1, DOF_Serie:])
            loss_DiffRelation = torch.tensor([0]).to(ComDevice)

            FirstDerivative_loss = myloss(Derivative_Data[:, :, :DOF_Serie],
                                          y[:, 1:-1, DOF_Serie:2 * DOF_Serie]).detach().to(
                torch.device('cpu')).numpy()
            SecondDerivative_loss = myloss(Derivative_Data[:, :, DOF_Serie:], y[:, 1:-1, 2 * DOF_Serie:]).detach().to(
                torch.device('cpu')).numpy()
            total_loss = data_loss + FirstDerivative_loss + SecondDerivative_loss

            # total_loss = data_loss * data_weight
            train_pino_test += loss_f.item()
            train_loss_test += total_loss.item()
            DiffRelation_loss_test += loss_DiffRelation.item()
            SignalLoss_test += data_loss
            FirstDerivativeLoss_test += FirstDerivative_loss
            SecondDerivativeLoss_test += SecondDerivative_loss

        scheduler.step()
        data_l2 /= len(train_loader)
        train_pino /= len(train_loader)
        virtual_pino /= len(train_loader)
        train_loss /= len(train_loader)
        DiffRelation_loss /= len(train_loader)
        Magnitude_test /= len(train_loader)

        train_pino_test /= len(test_loader)
        train_loss_test /= len(test_loader)
        DiffRelation_loss_test /= len(test_loader)
        SignalLoss_test /= len(test_loader)
        FirstDerivativeLoss_test /= len(test_loader)
        SecondDerivativeLoss_test /= len(test_loader)
        # if use_tqdm:
        #     pbar.set_description(
        #         (
        #             f'Epoch {e}, train loss: {train_loss:.5f} '
        #             f'train f error: {train_pino * f_weight:.5f}; '
        #             f'data l2 error: {data_l2:.5f}; '
        #             f'Magnitude Situation: {Magnitude:.1f}'
        #         )
        #     )
        # if wandb and log:
        #     wandb.log(
        #         {
        #             'Train f error': train_pino,
        #             'Train L2 error': data_l2,
        #             'Train loss': train_loss,
        #         }
        #     )
        LossRecord[Index, 0:7] = [DiffRelation_loss, train_pino * f_weight, virtual_pino * fv_weight, data_l2,
                                  SignalLoss_test, FirstDerivativeLoss_test, SecondDerivativeLoss_test]
        LossRecord[Index, 7:] = Magnitude_test
        Index += 1
        if e % 250 == 0:
            save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                            model, optimizer)
        if e % 2 == 0:
            if PerformanceSwitch == 'On':
                np.savetxt('checkpoints/' + config['train']['save_dir'] + '/' + 'Performance' + str(Index) + '.txt',
                           torch.cat([out[0, :, :], y[0, :, :DOF_Serie]], dim=-1).detach().to(
                               torch.device('cpu')).numpy())
            np.savetxt('checkpoints/' + config['train']['save_dir'] + '/' + config['train']['LossFileName'] + '.txt',
                       LossRecord)
        if e == config['train']['epochs'] - 1:
            np.savetxt('checkpoints/' + config['train']['save_dir'] + '/' + config['train']['LossFileName'] + '.txt',
                       LossRecord)
        if Index >= 275:
            if np.sum(LossRecord[Index, -3:]) < np.sum(LossRecord[Index - 1, -3:]):
                save_checkpoint(config['train']['save_dir'],
                                config['train']['save_name'].replace('.pt', 'Best_On_Test.pt'),
                                model, optimizer)
        if Index >= 50:
            if e % 25 == 0:
                save_checkpoint(config['train']['save_dir'],
                                config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                                model, optimizer)
        print('train loss:{}; f loss:{}; fv loss:{}; DiffRelation loss:{}, data loss:{}'.format(train_loss,
                                                                                                train_pino * f_weight,
                                                                                                virtual_pino * fv_weight,
                                                                                                1.0 * DiffRelation_loss,
                                                                                                data_l2 * data_weight))
        print('test loss:{}; Signal loss:{}; FirstDerivative loss:{}, SecondDerivative loss:{}'.format(train_loss_test,
                                                                                                       SignalLoss_test,
                                                                                                       FirstDerivativeLoss_test,
                                                                                                       SecondDerivativeLoss_test))
        # print('Magnitude of PDE loss for different equations:', Magnitude_test)
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    print('Done!')


def train_AMA(model,
              train_loader, test_loader,
              optimizer, scheduler,
              config,
              ToOneV,
              inputDim, outputDim,
              rank=0, log=False,
              project='PINO-AMA',
              group='default',
              tags=['default'],
              use_tqdm=False
              ):
    """
    Felexible model Example No.1
    Mapping only the solution (displacement)
    Offering the Ground Truth of the first and second derivative
    Using PDE restrain (derivative coming from the solution)
    """
    if rank == 0 and wandb and log:
        run = wandb.init(project=project,
                         entity='hzzheng-pino',
                         group=group,
                         config=config,
                         tags=tags, reinit=True,
                         settings=wandb.Settings(start_method="fork"))

    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    ic_weight = config['train']['ic_loss']
    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    # Record of loss functions
    LossRecord = np.zeros((config['train']['epochs'] + 1, 4))
    Index = 0

    for e in pbar:
        model.train()
        train_pino = 0.0
        data_l2 = 0.0
        train_loss = 0.0
        Magnitude = 0.0
        DiffRelation_loss = 0.0

        train_pino_test = 0.0
        data_l2_test = 0.0
        train_loss_test = 0.0
        SignalLoss_test = 0.0

        for x, y, Det_weights in train_loader:
            x, y = x.to(rank), y.to(rank)
            DOF_para = 3
            DOF_mode = y.size(-1) - DOF_para
            # ManualSpy
            # print('Shape of input is{}; output is{} and model output is{}'.format(x.shape, y.shape, model(x).shape))
            out = model(x).reshape(y.size(0), outputDim)
            data_loss = myloss(out, y)  # Pure data losses
            # Direct PDE loss of the signal
            loss_f = AMA_PINO_loss(out, x, ToOneV, DOF_para, Det_weights)
            # loss_f = torch.tensor([0]).to(torch.device('cuda:0'))
            total_loss = data_loss * data_weight + f_weight * loss_f

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            data_l2 += data_loss.item()
            train_pino += loss_f.item()
            train_loss += total_loss.item()

        for x, y, Det_weights_test in test_loader:
            x, y = x.to(rank), y.to(rank)
            # ManualSpy
            out = model(x).reshape(y.size(0), outputDim)
            data_loss = myloss(out, y).detach().to(torch.device('cpu'))
            # Direct PDE loss of the signal
            loss_f = AMA_PINO_loss(out, x, ToOneV, DOF_para, Det_weights_test)
            # loss_f = torch.tensor([0]).to(torch.device('cuda:0'))

            total_loss = data_loss + f_weight * loss_f

            # total_loss = data_loss * data_weight
            train_pino_test += loss_f.item()
            train_loss_test += total_loss.item()
            SignalLoss_test += data_loss

        scheduler.step()
        data_l2 /= len(train_loader)
        train_pino /= len(train_loader)
        train_loss /= len(train_loader)

        train_pino_test /= len(test_loader)
        train_loss_test /= len(test_loader)
        SignalLoss_test /= len(test_loader)
        # if use_tqdm:
        #     pbar.set_description(
        #         (
        #             f'Epoch {e}, train loss: {train_loss:.5f} '
        #             f'train f error: {train_pino * f_weight:.5f}; '
        #             f'data l2 error: {data_l2:.5f}; '
        #             f'Magnitude Situation: {Magnitude:.1f}'
        #         )
        #     )
        # if wandb and log:
        #     wandb.log(
        #         {
        #             'Train f error': train_pino,
        #             'Train L2 error': data_l2,
        #             'Train loss': train_loss,
        #         }
        #     )
        LossRecord[Index, 0:4] = [train_pino * f_weight, data_l2,
                                  SignalLoss_test, train_pino_test]
        Index += 1
        if e % 250 == 0:
            save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                            model, optimizer)
        if e % 2 == 0:
            np.savetxt('checkpoints/' + config['train']['save_dir'] + '/' + 'Performance' + str(Index) + '.txt',
                       torch.cat([out[0, :], y[0, :]], dim=-1).detach().to(
                           torch.device('cpu')).numpy())
            np.savetxt('checkpoints/' + config['train']['save_dir'] + '/LossHistory.txt', LossRecord)
        # if Index >= 50:
        #     if np.sum(LossRecord[Index, -2:]) < np.sum(LossRecord[Index - 1, -2:]):
        #         save_checkpoint(config['train']['save_dir'],
        #                         config['train']['save_name'].replace('.pt', 'Best_On_Test.pt'),
        #                         model, optimizer)
        # if Index >= 275:
        #     if e % 2 == 0:
        #         save_checkpoint(config['train']['save_dir'],
        #                         config['train']['save_name'].replace('.pt', f'_{e}.pt'),
        #                         model, optimizer)
        print('train loss:{}; f loss:{}; data loss:{}'.format(train_loss, train_pino * f_weight, data_l2))
        print(
            'test loss:{}; test_f loss:{}; test_data loss:{}'.format(train_loss_test, train_pino_test, SignalLoss_test))
        # print('Magnitude of PDE loss for different equations:', Magnitude_test)
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'], model, optimizer)
    print('Done!')


def train_AMA_WithVirtual(model,
                          train_loader, test_loader, virtual_loader,
                          optimizer, scheduler,
                          config,
                          ToOneV,
                          inputDim, outputDim,
                          rank=0, log=False,
                          project='PINO-AMA',
                          group='default',
                          tags=['default'],
                          use_tqdm=False
                          ):
    """
    Felexible model Example No.1
    Mapping only the solution (displacement)
    Offering the Ground Truth of the first and second derivative
    Using PDE restrain (derivative coming from the solution)
    """
    if rank == 0 and wandb and log:
        run = wandb.init(project=project,
                         entity='hzzheng-pino',
                         group=group,
                         config=config,
                         tags=tags, reinit=True,
                         settings=wandb.Settings(start_method="fork"))

    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    f_virtual_weight = config['train']['f_virtual_loss']
    ic_weight = config['train']['ic_loss']
    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    # Record of loss functions
    LossRecord = np.zeros((config['train']['epochs'] + 1, 5))
    Index = 0

    for e in pbar:
        model.train()
        train_pino = 0.0
        virtual_pino = 0.0
        data_l2 = 0.0
        train_loss = 0.0
        Magnitude = 0.0
        DiffRelation_loss = 0.0

        train_pino_test = 0.0
        data_l2_test = 0.0
        train_loss_test = 0.0
        SignalLoss_test = 0.0

        for x, y, Det_weights in train_loader:
            x, y = x.to(rank), y.to(rank)
            DOF_para = 3
            DOF_mode = y.size(-1) - DOF_para
            # ManualSpy
            # print('Shape of input is{}; output is{} and model output is{}'.format(x.shape, y.shape, model(x).shape))
            out = model(x).reshape(y.size(0), outputDim)
            data_loss = myloss(out, y)  # Pure data losses
            # Direct PDE loss of the signal
            loss_f = AMA_PINO_loss(out, x, ToOneV, DOF_para, Det_weights)

            x_virtual = next(iter(virtual_loader))[0].to(rank)
            out_virtual = model(x_virtual).reshape(x_virtual.size(0), outputDim)
            virtual_weights = torch.Tensor([1.2394e9, 1.1857e11, 8.6890e13, 1.7819e14, 2.5280e18, 2.4682e20]). \
                unsqueeze(0).double().repeat_interleave(x_virtual.size(0), dim=0)
            loss_f_virtual = AMA_PINO_loss(out_virtual, x_virtual, ToOneV, DOF_para, virtual_weights)
            # loss_f = torch.tensor([0]).to(torch.device('cuda:0'))
            total_loss = data_loss * data_weight + f_weight * loss_f + f_virtual_weight * loss_f_virtual

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            data_l2 += data_loss.item()
            train_pino += loss_f.item()
            virtual_pino += loss_f_virtual.item()
            train_loss += total_loss.item()

        for x, y, Det_weights_test in test_loader:
            x, y = x.to(rank), y.to(rank)
            # ManualSpy
            out = model(x).reshape(y.size(0), outputDim)
            data_loss = myloss(out, y).detach().to(torch.device('cpu'))
            # Direct PDE loss of the signal
            loss_f = AMA_PINO_loss(out, x, ToOneV, DOF_para, Det_weights_test)
            # loss_f = torch.tensor([0]).to(torch.device('cuda:0'))

            total_loss = data_loss + 0.0000000001 * loss_f

            # total_loss = data_loss * data_weight
            train_pino_test += loss_f.item()
            train_loss_test += total_loss.item()
            SignalLoss_test += data_loss

        scheduler.step()
        data_l2 /= len(train_loader)
        train_pino /= len(train_loader)
        virtual_pino /= len(train_loader)
        train_loss /= len(train_loader)

        train_pino_test /= len(test_loader)
        train_loss_test /= len(test_loader)
        SignalLoss_test /= len(test_loader)
        # if use_tqdm:
        #     pbar.set_description(
        #         (
        #             f'Epoch {e}, train loss: {train_loss:.5f} '
        #             f'train f error: {train_pino * f_weight:.5f}; '
        #             f'data l2 error: {data_l2:.5f}; '
        #             f'Magnitude Situation: {Magnitude:.1f}'
        #         )
        #     )
        # if wandb and log:
        #     wandb.log(
        #         {
        #             'Train f error': train_pino,
        #             'Train L2 error': data_l2,
        #             'Train loss': train_loss,
        #         }
        #     )
        LossRecord[Index, 0:5] = [train_pino * f_weight, f_virtual_weight * virtual_pino, data_l2,
                                  SignalLoss_test, train_pino_test]
        Index += 1
        if e % 250 == 0:
            save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                            model, optimizer)
        if e % 2 == 0:
            np.savetxt('checkpoints/' + config['train']['save_dir'] + '/' + 'Performance' + str(Index) + '.txt',
                       torch.cat([out[0, :], y[0, :]], dim=-1).detach().to(
                           torch.device('cpu')).numpy())
            np.savetxt('checkpoints/' + config['train']['save_dir'] + '/LossHistory.txt', LossRecord)
        # if Index >= 50:
        #     if np.sum(LossRecord[Index, -2:]) < np.sum(LossRecord[Index - 1, -2:]):
        #         save_checkpoint(config['train']['save_dir'],
        #                         config['train']['save_name'].replace('.pt', 'Best_On_Test.pt'),
        #                         model, optimizer)
        # if Index >= 275:
        #     if e % 2 == 0:
        #         save_checkpoint(config['train']['save_dir'],
        #                         config['train']['save_name'].replace('.pt', f'_{e}.pt'),
        #                         model, optimizer)
        print('train loss:{}; f loss:{}; virtual f loss:{}, data loss:{}'.format(train_loss, train_pino * f_weight,
                                                                                 virtual_pino * f_virtual_weight,
                                                                                 data_l2))
        print(
            'test loss:{}; test_f loss:{}; test_data loss:{}'.format(train_loss_test, train_pino_test, SignalLoss_test))
        # print('Magnitude of PDE loss for different equations:', Magnitude_test)
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'], model, optimizer)
    print('Done!')


def train_VTCD(model,
               train_loader, test_loader,
               optimizer, scheduler,
               config,
               ToOneV,
               inputDim, outputDim, D,
               rank=0, log=False,
               project='PINO-VTCD',
               group='default',
               tags=['default'],
               use_tqdm=False
               ):
    """
    Light Variant Number one
    Mapping only the solution (displacement)
    Offering the Ground Truth of the first and second derivative
    Using PDE restrain (derivative coming from the solution)
    """
    if rank == 0 and wandb and log:
        run = wandb.init(project=project,
                         entity='hzzheng-pino',
                         group=group,
                         config=config,
                         tags=tags, reinit=True,
                         settings=wandb.Settings(start_method="fork"))

    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    ic_weight = config['train']['ic_loss']
    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    # Record of loss functions
    LossRecord = np.zeros((config['train']['epochs'] + 1, 26))
    Index = 0

    for e in pbar:
        model.train()
        train_pino = 0.0
        data_l2 = 0.0
        train_loss = 0.0
        Magnitude = 0.0
        Boundary_loss = 0.0
        DiffRelation_loss = 0.0

        train_pino_test = 0.0
        data_l2_test = 0.0
        train_loss_test = 0.0
        Magnitude_test = np.zeros((1, 20))
        Boundary_loss_test = 0.0
        DiffRelation_loss_test = 0.0
        SignalLoss_test = 0.0
        FirstDerivativeLoss_test = 0.0
        SecondDerivativeLoss_test = 0.0

        for x, y, PDE_weights in train_loader:
            x, y = x.to(rank), y.to(rank)
            # ManualSpy
            # print('Shape of input is{}; output is{} and model output is{}'.format(x.shape, y.shape, model(x).shape))
            out = model(x).reshape(y.size(0), y.size(1), outputDim)
            data_loss = myloss(out, torch.cat([y[:, :, :10], y[:, :, -4:]], dim=-1))  # Pure dataloss
            # Boundary loss for the signal
            # loss_BC = VTCD_BC_loss_WithRail(out, y)
            loss_BC = torch.tensor([0.0]).to(torch.device('cuda:0'))
            # Direct PDE loss of the signal
            loss_f, Du_Magnitude, Derivative_Data = VTCD_PINO_loss(out, x, PDE_weights, ToOneV, inputDim, outputDim, D)
            loss_f = torch.tensor([0]).to(torch.device('cuda:0'))
            # Du_Magnitude = torch.tensor([0])
            # Diff Relation loss for the signal pack
            criterion = torch.nn.L1Loss()
            # loss_DiffRelation = criterion(Derivative_Data, y[:, 1:-1, 10:-4])
            loss_DiffRelation = myloss(Derivative_Data[:, :, :10], y[:, 1:-1, 10:20]) + myloss(
                Derivative_Data[:, :, 10:], y[:, 1:-1, 20:-4])
            # loss_DiffRelation = torch.tensor([0]).to(torch.device('cuda:0'))

            # total_loss = data_loss * data_weight + 1.0 * loss_BC
            total_loss = loss_f * f_weight + data_loss * data_weight + 1.0 * loss_BC + 1.0 * loss_DiffRelation

            # total_loss = data_loss * data_weight

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            data_l2 += data_loss.item()
            train_pino += loss_f.item()
            train_loss += total_loss.item()
            Boundary_loss += loss_BC.item()
            DiffRelation_loss += loss_DiffRelation.item()
            Magnitude_test += Du_Magnitude.detach().to(torch.device('cpu')).numpy()
            # Magnitude = (torch.max(Du_Magnitude) / torch.min(Du_Magnitude)).item()

        for x, y in test_loader:
            x, y = x.to(rank), y.to(rank)
            # ManualSpy
            # print('Shape of input is{}; output is{} and model output is{}'.format(x.shape, y.shape, model(x).shape))
            out = model(x).reshape(y.size(0), y.size(1), outputDim)
            data_loss = myloss(out, torch.cat([y[:, :, :10], y[:, :, -4:]], dim=-1)).detach().to(
                torch.device('cpu')).numpy()  # Pure dataloss
            # Boundary loss for the signal
            # loss_BC = VTCD_BC_loss_WithRail(out, y)
            loss_BC = torch.tensor([0.0]).to(torch.device('cuda:0'))
            # Direct PDE loss of the signal
            loss_f, Dummy, Derivative_Data = VTCD_PINO_loss_Variant1(out, x, ToOneV, inputDim, outputDim, D)
            # loss_f = torch.tensor([0]).to(torch.device('cuda:0'))
            # Du_Magnitude = torch.tensor([0])
            # Diff Relation loss for the signal pack
            # Diff_Weight = torch.ones(20).to(torch.device('cuda:0'))
            # Diff_Weight[0], Diff_Weight[10] = 2, 2
            # loss_DiffRelation = myloss(Derivative_Data, Diff_Weight * y[:, 1:-1, 10:-4])
            loss_DiffRelation = torch.tensor([0]).to(torch.device('cuda:0'))

            # total_loss = data_loss * data_weight + 1.0 * loss_BC
            # total_loss = loss_f * f_weight + data_loss * data_weight + 1.0 * loss_BC + loss_DiffRelation
            FirstDerivative_loss = myloss(Derivative_Data[:, :, :10], y[:, 1:-1, 10:20]).detach().to(
                torch.device('cpu')).numpy()
            SecondDerivative_loss = myloss(Derivative_Data[:, :, 10:], y[:, 1:-1, 20:-4]).detach().to(
                torch.device('cpu')).numpy()
            total_loss = data_loss + FirstDerivative_loss + SecondDerivative_loss

            # total_loss = data_loss * data_weight
            train_pino_test += loss_f.item()
            train_loss_test += total_loss.item()
            Boundary_loss_test += loss_BC.item()
            DiffRelation_loss_test += loss_DiffRelation.item()
            SignalLoss_test += data_loss
            FirstDerivativeLoss_test += FirstDerivative_loss
            SecondDerivativeLoss_test += SecondDerivative_loss

        scheduler.step()
        data_l2 /= len(train_loader)
        train_pino /= len(train_loader)
        train_loss /= len(train_loader)
        Boundary_loss /= len(train_loader)
        DiffRelation_loss /= len(train_loader)
        Magnitude_test /= len(train_loader)

        train_pino_test /= len(test_loader)
        train_loss_test /= len(test_loader)
        Boundary_loss_test /= len(test_loader)
        DiffRelation_loss_test /= len(test_loader)
        SignalLoss_test /= len(test_loader)
        FirstDerivativeLoss_test /= len(test_loader)
        SecondDerivativeLoss_test /= len(test_loader)
        # if use_tqdm:
        #     pbar.set_description(
        #         (
        #             f'Epoch {e}, train loss: {train_loss:.5f} '
        #             f'train f error: {train_pino * f_weight:.5f}; '
        #             f'data l2 error: {data_l2:.5f}; '
        #             f'Magnitude Situation: {Magnitude:.1f}'
        #         )
        #     )
        # if wandb and log:
        #     wandb.log(
        #         {
        #             'Train f error': train_pino,
        #             'Train L2 error': data_l2,
        #             'Train loss': train_loss,
        #         }
        #     )
        LossRecord[Index, 0:6] = [DiffRelation_loss, train_pino * f_weight, data_l2,
                                  SignalLoss_test, FirstDerivativeLoss_test, SecondDerivativeLoss_test]
        LossRecord[Index, 6:] = Magnitude_test
        Index += 1
        if e % 250 == 0:
            save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                            model, optimizer)
        if e % 2 == 0:
            np.savetxt('checkpoints/' + config['train']['save_dir'] + '/' + 'Performance' + str(Index) + '.txt',
                       torch.cat([out[0, :, :], torch.cat([y[0, :, :10], y[0, :, -4:]], dim=-1)], dim=-1).detach().to(
                           torch.device('cpu')).numpy())
            np.savetxt('checkpoints/' + config['train']['save_dir'] + '/LossHistory.txt', LossRecord)
        if Index >= 50:
            if np.sum(LossRecord[Index, -3:]) < np.sum(LossRecord[Index - 1, -3:]):
                save_checkpoint(config['train']['save_dir'],
                                config['train']['save_name'].replace('.pt', 'Best_On_Test.pt'),
                                model, optimizer)
        if Index >= 275:
            if e % 2 == 0:
                save_checkpoint(config['train']['save_dir'],
                                config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                                model, optimizer)
        print('train loss:{}; f loss:{}; DiffRelation loss:{}, data loss:{}'.format(train_loss,
                                                                                    train_pino * f_weight,
                                                                                    1.0 * DiffRelation_loss,
                                                                                    data_l2))
        print('test loss:{}; Signal loss:{}; FirstDerivative loss:{}, SecondDerivative loss:{}'.format(train_loss_test,
                                                                                                       SignalLoss_test,
                                                                                                       FirstDerivativeLoss_test,
                                                                                                       SecondDerivativeLoss_test))
        # print('Magnitude of PDE loss for different equations:', Magnitude_test)
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    print('Done!')


def train_VTCD_WithVirtual(model,
                           train_loader, test_loader, virtual_loader, PDE_weights_virtual,
                           optimizer, scheduler,
                           config,
                           ToOneV,
                           inputDim, outputDim, D, ComDevice,
                           rank=0, log=False,
                           project='PINO-VTCD',
                           group='default',
                           tags=['default'],
                           use_tqdm=False
                           ):
    """
    Light Variant Number one
    Mapping only the solution (displacement)
    Offering the Ground Truth of the first and second derivative
    Using PDE restrain (derivative coming from the solution)
    """
    if rank == 0 and wandb and log:
        run = wandb.init(project=project,
                         entity='hzzheng-pino',
                         group=group,
                         config=config,
                         tags=tags, reinit=True,
                         settings=wandb.Settings(start_method="fork"))

    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    fv_weight = config['train']['fv_loss']
    diff_weight = config['train']['diff_loss']
    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    # Record of loss functions
    LossRecord = np.zeros((config['train']['epochs'] + 1, 7 + 10 * 2))
    Index = 0

    for e in pbar:
        model.train()
        train_pino = 0.0
        virtual_pino = 0.0
        data_l2 = 0.0
        train_loss = 0.0
        Magnitude = 0.0
        Boundary_loss = 0.0
        DiffRelation_loss = 0.0

        train_pino_test = 0.0
        data_l2_test = 0.0
        train_loss_test = 0.0
        Magnitude_test = np.zeros((1, 20))
        Boundary_loss_test = 0.0
        DiffRelation_loss_test = 0.0
        SignalLoss_test = 0.0
        FirstDerivativeLoss_test = 0.0
        SecondDerivativeLoss_test = 0.0
        VirtualSwitch = config['data']['VirtualSwitch']
        PerformanceSwitch = config['data']['PerformanceSwitch']

        for x, y, PDE_weights in train_loader:
            x, y = x.to(ComDevice), y.to(ComDevice)
            # ManualSpy
            # print('Shape of input is{}; output is{} and model output is{}'.format(x.shape, y.shape, model(x).shape))
            out = model(x).reshape(y.size(0), y.size(1), outputDim)
            data_loss = myloss(out, torch.cat([y[:, :, :10], y[:, :, -4:]], dim=-1))  # Pure dataloss
            # Boundary loss for the signal
            # loss_BC = VTCD_BC_loss_WithRail(out, y)
            loss_BC = torch.tensor([0.0]).to(torch.device('cuda:0'))
            # Direct PDE loss of the signal
            loss_f, Du_Magnitude, Derivative_Data = VTCD_PINO_loss(out, x, PDE_weights, ToOneV, inputDim, outputDim, D,
                                                                   ComDevice)
            # loss_f = torch.tensor([0]).to(torch.device('cuda:0'))
            # Du_Magnitude = torch.tensor([0])
            # Diff Relation loss for the signal pack
            criterion = torch.nn.L1Loss()
            # loss_DiffRelation = criterion(Derivative_Data, y[:, 1:-1, 10:-4])
            loss_DiffRelation = myloss(Derivative_Data[:, :, :10], y[:, 1:-1, 10:20]) + myloss(
                Derivative_Data[:, :, 10:], y[:, 1:-1, 20:-4])
            # loss_DiffRelation = torch.tensor([0]).to(torch.device('cuda:0'))

            # Harvest gradients on Virtual Data
            if VirtualSwitch == 'On':
                x_virtual = next(iter(virtual_loader))[0].to(ComDevice)
                out_virtual = model(x_virtual).reshape(x_virtual.size(0), y.size(1), outputDim)
                virtual_weights = PDE_weights_virtual.double().repeat_interleave(x_virtual.size(0), dim=0)
                loss_f_virtual, _, _ = VTCD_PINO_loss(out_virtual, x_virtual, virtual_weights, ToOneV, inputDim,
                                                      outputDim, D, ComDevice)
            else:
                loss_f_virtual = torch.tensor([0]).to(ComDevice)
            total_loss = loss_f_virtual * fv_weight + data_loss * data_weight + loss_DiffRelation * diff_weight + loss_f * f_weight

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            data_l2 += data_loss.item()
            train_pino += loss_f.item()
            virtual_pino += loss_f_virtual.item()
            train_loss += total_loss.item()
            Boundary_loss += loss_BC.item()
            DiffRelation_loss += loss_DiffRelation.item()
            Magnitude_test += Du_Magnitude.detach().to(torch.device('cpu')).numpy()
            # Magnitude = (torch.max(Du_Magnitude) / torch.min(Du_Magnitude)).item()

        for x, y in test_loader:
            x, y = x.to(ComDevice), y.to(ComDevice)
            # ManualSpy
            # print('Shape of input is{}; output is{} and model output is{}'.format(x.shape, y.shape, model(x).shape))
            out = model(x).reshape(y.size(0), y.size(1), outputDim)
            data_loss = myloss(out, torch.cat([y[:, :, :10], y[:, :, -4:]], dim=-1)).detach().to(
                torch.device('cpu')).numpy()  # Pure dataloss
            # Boundary loss for the signal
            # loss_BC = VTCD_BC_loss_WithRail(out, y)
            loss_BC = torch.tensor([0.0]).to(torch.device('cuda:0'))
            # Test operation: Use average weights for PDE loss computation
            test_weights = PDE_weights_virtual.double().repeat_interleave(x.size(0), dim=0)
            loss_f, _, Derivative_Data = VTCD_PINO_loss(out, x, test_weights, ToOneV, inputDim, outputDim, D, ComDevice)
            # loss_f = torch.tensor([0]).to(torch.device('cuda:0'))
            # Du_Magnitude = torch.tensor([0])
            # Diff Relation loss for the signal pack
            # Diff_Weight = torch.ones(20).to(torch.device('cuda:0'))
            # Diff_Weight[0], Diff_Weight[10] = 2, 2
            # loss_DiffRelation = myloss(Derivative_Data, Diff_Weight * y[:, 1:-1, 10:-4])
            loss_DiffRelation = torch.tensor([0]).to(ComDevice)

            # total_loss = data_loss * data_weight + 1.0 * loss_BC
            # total_loss = loss_f * f_weight + data_loss * data_weight + 1.0 * loss_BC + loss_DiffRelation
            FirstDerivative_loss = myloss(Derivative_Data[:, :, :10], y[:, 1:-1, 10:20]).detach().to(
                torch.device('cpu')).numpy()
            SecondDerivative_loss = myloss(Derivative_Data[:, :, 10:], y[:, 1:-1, 20:-4]).detach().to(
                torch.device('cpu')).numpy()
            total_loss = data_loss + FirstDerivative_loss + SecondDerivative_loss

            # total_loss = data_loss * data_weight
            train_pino_test += loss_f.item()
            train_loss_test += total_loss.item()
            Boundary_loss_test += loss_BC.item()
            DiffRelation_loss_test += loss_DiffRelation.item()
            SignalLoss_test += data_loss
            FirstDerivativeLoss_test += FirstDerivative_loss
            SecondDerivativeLoss_test += SecondDerivative_loss

        scheduler.step()
        data_l2 /= len(train_loader)
        train_pino /= len(train_loader)
        virtual_pino /= len(train_loader)
        train_loss /= len(train_loader)
        Boundary_loss /= len(train_loader)
        DiffRelation_loss /= len(train_loader)
        Magnitude_test /= len(train_loader)

        train_pino_test /= len(test_loader)
        train_loss_test /= len(test_loader)
        Boundary_loss_test /= len(test_loader)
        DiffRelation_loss_test /= len(test_loader)
        SignalLoss_test /= len(test_loader)
        FirstDerivativeLoss_test /= len(test_loader)
        SecondDerivativeLoss_test /= len(test_loader)
        # if use_tqdm:
        #     pbar.set_description(
        #         (
        #             f'Epoch {e}, train loss: {train_loss:.5f} '
        #             f'train f error: {train_pino * f_weight:.5f}; '
        #             f'data l2 error: {data_l2:.5f}; '
        #             f'Magnitude Situation: {Magnitude:.1f}'
        #         )
        #     )
        # if wandb and log:
        #     wandb.log(
        #         {
        #             'Train f error': train_pino,
        #             'Train L2 error': data_l2,
        #             'Train loss': train_loss,
        #         }
        #     )
        LossRecord[Index, 0:7] = [DiffRelation_loss, train_pino * f_weight, virtual_pino * fv_weight, data_l2,
                                  SignalLoss_test, FirstDerivativeLoss_test, SecondDerivativeLoss_test]
        LossRecord[Index, 7:] = Magnitude_test
        Index += 1
        if e % 250 == 0:
            save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                            model, optimizer)
        if e % 2 == 0:
            if PerformanceSwitch == 'On':
                np.savetxt('checkpoints/' + config['train']['save_dir'] + '/' + 'Performance' + str(Index) + '.txt',
                           torch.cat([out[0, :, :], torch.cat([y[0, :, :10], y[0, :, -4:]], dim=-1)],
                                     dim=-1).detach().to(
                               torch.device('cpu')).numpy())
            np.savetxt('checkpoints/' + config['train']['save_dir'] + '/LossHistory.txt', LossRecord)
        if Index >= 250:
            if np.sum(LossRecord[Index, -3:]) < np.sum(LossRecord[Index - 1, -3:]):
                save_checkpoint(config['train']['save_dir'],
                                config['train']['save_name'].replace('.pt', 'Best_On_Test.pt'),
                                model, optimizer)
        if Index >= 250:
            if e % 2 == 0:
                save_checkpoint(config['train']['save_dir'],
                                config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                                model, optimizer)
        print('train loss:{}; f loss:{}; fv loss:{}; DiffRelation loss:{}, data loss:{}'.format(train_loss,
                                                                                                train_pino * f_weight,
                                                                                                virtual_pino * fv_weight,
                                                                                                diff_weight * DiffRelation_loss,
                                                                                                data_l2 * data_weight))
        print('test loss:{}; Signal loss:{}; FirstDerivative loss:{}, SecondDerivative loss:{}'.format(train_loss_test,
                                                                                                       SignalLoss_test,
                                                                                                       FirstDerivativeLoss_test,
                                                                                                       SecondDerivativeLoss_test))
        # print('Magnitude of PDE loss for different equations:', Magnitude_test)
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    print('Done!')


def train_BSA_WithVirtual(model,
                          train_loader, test_loader, virtual_loader, PDE_weights_virtual,
                          optimizer, scheduler,
                          config,
                          ToOneV,
                          W2_CX, W2_CY, W2_CZ,
                          Eigens2, TrackDOFs, Nloc,
                          inputDim, outputDim, D, ComDevice,
                          rank=0, log=False,
                          project='PINO-BSA',
                          group='default',
                          tags=['default'],
                          use_tqdm=False
                          ):
    """
    Felexible model Example No.3
    Mapping only the solution (displacement)
    Offering the Ground Truth of the first and second derivative
    Using PDE restrain (derivative coming from the solution)
    """
    if rank == 0 and wandb and log:
        run = wandb.init(project=project,
                         entity='hzzheng-pino',
                         group=group,
                         config=config,
                         tags=tags, reinit=True,
                         settings=wandb.Settings(start_method="fork"))

    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    fv_weight = config['train']['fv_loss']
    diff_weight = config['train']['diff_loss']
    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    # Record of loss functions
    LossRecord = np.zeros((config['train']['epochs'] + 1, 7 + 200 * 2))
    Index = 0

    for e in pbar:
        model.train()
        train_pino = 0.0
        virtual_pino = 0.0
        data_l2 = 0.0
        train_loss = 0.0
        Magnitude = 0.0
        DiffRelation_loss = 0.0

        train_pino_test = 0.0
        data_l2_test = 0.0
        train_loss_test = 0.0
        Magnitude_test = np.zeros((1, 2 * 200))
        DiffRelation_loss_test = 0.0
        SignalLoss_test = 0.0
        FirstDerivativeLoss_test = 0.0
        SecondDerivativeLoss_test = 0.0
        VirtualSwitch = config['data']['VirtualSwitch']
        PerformanceSwitch = config['data']['PerformanceSwitch']

        for x, y, PDE_weights in train_loader:
            x, y = x.to(ComDevice), y.to(ComDevice)
            DOF_exc = 6
            DOF_flex = int((1 / 3) * y.shape[-1])

            # ManualSpy
            # print('Shape of input is{}; output is{} and model output is{}'.format(x.shape, y.shape, model(x).shape))
            out = model(x).reshape(y.size(0), y.size(1), outputDim)
            data_loss = myloss(out, y[:, :, :DOF_flex])  # Pure data losses
            # Direct PDE loss of the signal
            loss_f, Du_Magnitude, Derivative_Data = BSA_PINO_loss(out, x, PDE_weights, ToOneV, inputDim, outputDim, D,
                                                                  DOF_exc, DOF_flex, W2_CX, W2_CY, W2_CZ, Eigens2,
                                                                  TrackDOFs,
                                                                  Nloc, ComDevice, switch=1)

            if VirtualSwitch == 'On':
                x_virtual = next(iter(virtual_loader))[0].to(ComDevice)
                # print(model(x_virtual).shape, x_virtual.shape)
                out_virtual = model(x_virtual)
                virtual_weights = PDE_weights_virtual.double().repeat_interleave(x_virtual.size(0), dim=0)
                loss_f_virtual, _, _ = BSA_PINO_loss(out_virtual, x_virtual, virtual_weights, ToOneV, inputDim,
                                                     outputDim, D, DOF_exc, DOF_flex, W2_CX, W2_CY, W2_CZ, Eigens2,
                                                     TrackDOFs,
                                                     Nloc, ComDevice, switch=0)
            else:
                loss_f_virtual = torch.tensor([0]).to(ComDevice)
            # loss_f = torch.tensor([0]).to(torch.device('cuda:0'))
            # Du_Magnitude = torch.tensor([0])
            # Diff Relation loss for the signal pack
            criterion = torch.nn.L1Loss()
            # loss_DiffRelation = criterion(Derivative_Data, y[:, 1:-1, DOF_Serie:])
            # loss_DiffRelation = myloss(Derivative_Data[:, :, :DOF_Serie], y[:, 1:-1, DOF_Serie: 2*DOF_Serie]) + myloss(Derivative_Data[:, :, DOF_Serie:], y[:, 1:-1, 2*DOF_Serie:])

            loss_DiffRelation = myloss(Derivative_Data, y[:, 1:-1, DOF_flex:])

            # loss_DiffRelation = torch.tensor([0]).to(torch.device('cuda:0'))

            total_loss = loss_f * f_weight + loss_f_virtual * fv_weight + data_loss * data_weight + loss_DiffRelation * diff_weight
            # total_loss = loss_f_virtual * fv_weight + data_loss * data_weight + loss_DiffRelation * diff_weight
            # total_loss = data_loss * data_weight + 1.0 * loss_DiffRelation

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            data_l2 += data_loss.item()
            train_pino += loss_f.item()
            virtual_pino += loss_f_virtual.item()
            train_loss += total_loss.item()
            DiffRelation_loss += loss_DiffRelation.item()
            Magnitude_test += Du_Magnitude.detach().to(torch.device('cpu')).numpy()
            # Magnitude = (torch.max(Du_Magnitude) / torch.min(Du_Magnitude)).item()

        for x, y, PDE_weights in test_loader:
            x, y = x.to(ComDevice), y.to(ComDevice)
            # ManualSpy
            out = model(x).reshape(y.size(0), y.size(1), outputDim)
            data_loss = myloss(out, y[:, :, :DOF_flex]).detach().to(torch.device('cpu')).numpy()
            # Direct PDE loss of the signal
            loss_f, Du_Magnitude, Derivative_Data = BSA_PINO_loss(out, x, PDE_weights, ToOneV, inputDim, outputDim, D,
                                                                  DOF_exc, DOF_flex, W2_CX, W2_CY, W2_CZ, Eigens2,
                                                                  TrackDOFs,
                                                                  Nloc, ComDevice, switch=1)
            # loss_f = torch.tensor([0]).to(torch.device('cuda:0'))
            # Du_Magnitude = torch.tensor([0])
            # Diff Relation loss for the signal pack
            # Diff_Weight = torch.ones(20).to(torch.device('cuda:0'))
            # Diff_Weight[0], Diff_Weight[10] = 2, 2
            # loss_DiffRelation = myloss(Derivative_Data, y[:, 1:-1, DOF_Serie:])
            loss_DiffRelation = torch.tensor([0]).to(ComDevice)

            FirstDerivative_loss = myloss(Derivative_Data[:, :, :DOF_flex],
                                          y[:, 1:-1, DOF_flex:2 * DOF_flex]).detach().to(torch.device('cpu')).numpy()
            SecondDerivative_loss = myloss(Derivative_Data[:, :, DOF_flex:], y[:, 1:-1, 2 * DOF_flex:]).detach().to(
                torch.device('cpu')).numpy()
            total_loss = data_loss + FirstDerivative_loss + SecondDerivative_loss

            # total_loss = data_loss * data_weight
            train_pino_test += loss_f.item()
            train_loss_test += total_loss.item()
            DiffRelation_loss_test += loss_DiffRelation.item()
            SignalLoss_test += data_loss
            FirstDerivativeLoss_test += FirstDerivative_loss
            SecondDerivativeLoss_test += SecondDerivative_loss

        scheduler.step()
        data_l2 /= len(train_loader)
        train_pino /= len(train_loader)
        virtual_pino /= len(train_loader)
        train_loss /= len(train_loader)
        DiffRelation_loss /= len(train_loader)
        Magnitude_test /= len(train_loader)

        train_pino_test /= len(test_loader)
        train_loss_test /= len(test_loader)
        DiffRelation_loss_test /= len(test_loader)
        SignalLoss_test /= len(test_loader)
        FirstDerivativeLoss_test /= len(test_loader)
        SecondDerivativeLoss_test /= len(test_loader)
        # if use_tqdm:
        #     pbar.set_description(
        #         (
        #             f'Epoch {e}, train loss: {train_loss:.5f} '
        #             f'train f error: {train_pino * f_weight:.5f}; '
        #             f'data l2 error: {data_l2:.5f}; '
        #             f'Magnitude Situation: {Magnitude:.1f}'
        #         )
        #     )
        # if wandb and log:
        #     wandb.log(
        #         {
        #             'Train f error': train_pino,
        #             'Train L2 error': data_l2,
        #             'Train loss': train_loss,
        #         }
        #     )
        LossRecord[Index, 0:7] = [DiffRelation_loss, train_pino * f_weight, virtual_pino * fv_weight, data_l2,
                                  SignalLoss_test, FirstDerivativeLoss_test, SecondDerivativeLoss_test]
        LossRecord[Index, 7:] = Magnitude_test
        Index += 1
        if e % 250 == 0:
            save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                            model, optimizer)
        if e % 2 == 0:
            if PerformanceSwitch == 'On':
                np.savetxt('checkpoints/' + config['train']['save_dir'] + '/' + 'Performance' + str(Index) + '.txt',
                           torch.cat([out[0, :, :], y[0, :, :DOF_flex]], dim=-1).detach().to(
                               torch.device('cpu')).numpy())
            np.savetxt('checkpoints/' + config['train']['save_dir'] + '/' + config['train']['LossFileName'] + '.txt',
                       LossRecord)
        if Index >= 475:
            if np.sum(LossRecord[Index, -3:]) < np.sum(LossRecord[Index - 1, -3:]):
                save_checkpoint(config['train']['save_dir'],
                                config['train']['save_name'].replace('.pt', 'Best_On_Test.pt'),
                                model, optimizer)
                np.savetxt('checkpoints/' + config['train']['save_dir'] + '/' + config['train']['LossFileName'] + '.txt', LossRecord)
        if Index >= 400:
            if e % 10 == 0:
                save_checkpoint(config['train']['save_dir'],
                                config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                                model, optimizer)
                np.savetxt('checkpoints/' + config['train']['save_dir'] + '/' + config['train']['LossFileName'] + '.txt', LossRecord)
        print('train loss:{}; f loss:{}; fv loss:{}; DiffRelation loss:{}, data loss:{}'.format(train_loss,
                                                                                                train_pino * f_weight,
                                                                                                virtual_pino * fv_weight,
                                                                                                1.0 * DiffRelation_loss,
                                                                                                data_l2 * data_weight))
        print('test loss:{}; Signal loss:{}; FirstDerivative loss:{}, SecondDerivative loss:{}'.format(train_loss_test,
                                                                                                       SignalLoss_test,
                                                                                                       FirstDerivativeLoss_test,
                                                                                                       SecondDerivativeLoss_test))
        # print('Magnitude of PDE loss for different equations:', Magnitude_test)
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    print('Done!')


def train_BSA_CompleteVirtual(model,
                              train_loader, test_loader, virtual_loader, PDE_weights_virtual,
                              optimizer, scheduler,
                              config,
                              ToOneV,
                              W2_CX, W2_CY, W2_CZ,
                              Eigens2, TrackDOFs, Nloc,
                              inputDim, outputDim, D, ComDevice,
                              rank=0, log=False,
                              project='PINO-BSA',
                              group='default',
                              tags=['default'],
                              use_tqdm=False
                              ):
    """
    Felexible model Example No.3 (With Complete Virtual Data)
    Mapping only the solution (displacement)
    Offering the Ground Truth of the first and second derivative
    Using PDE restrain (derivative coming from the solution)
    """
    if rank == 0 and wandb and log:
        run = wandb.init(project=project,
                         entity='hzzheng-pino',
                         group=group,
                         config=config,
                         tags=tags, reinit=True,
                         settings=wandb.Settings(start_method="fork"))

    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    fv_weight = config['train']['fv_loss']
    diff_weight = config['train']['diff_loss']
    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    # Record of loss functions
    LossRecord = np.zeros((config['train']['epochs'] + 1, 7 + 200 * 2))
    Index = 0

    for e in pbar:
        model.train()
        train_pino = 0.0
        virtual_pino = 0.0
        data_l2 = 0.0
        train_loss = 0.0
        Magnitude = 0.0
        DiffRelation_loss = 0.0

        train_pino_test = 0.0
        data_l2_test = 0.0
        train_loss_test = 0.0
        Magnitude_test = np.zeros((1, 2 * 200))
        DiffRelation_loss_test = 0.0
        SignalLoss_test = 0.0
        FirstDerivativeLoss_test = 0.0
        SecondDerivativeLoss_test = 0.0
        VirtualSwitch = config['data']['VirtualSwitch']
        PerformanceSwitch = config['data']['PerformanceSwitch']

        for x in virtual_loader:
            x_virtual = x[0].to(ComDevice)
            # print(model(x_virtual).shape, x_virtual.shape)
            out_virtual = model(x_virtual)
            DOF_exc = 6
            DOF_flex = 200
            virtual_weights = PDE_weights_virtual.double().repeat_interleave(x_virtual.size(0), dim=0)
            loss_f_virtual, _, _ = BSA_PINO_loss(out_virtual, x_virtual, virtual_weights, ToOneV, inputDim,
                                                 outputDim, D, DOF_exc, DOF_flex, W2_CX, W2_CY, W2_CZ, Eigens2,
                                                 TrackDOFs,
                                                 Nloc, ComDevice, switch=0)

            loss_f = torch.tensor([0]).to(torch.device('cuda:0'))

            data_loss = torch.tensor([0]).to(torch.device('cuda:0'))
            # Du_Magnitude = torch.tensor([0])
            # Diff Relation loss for the signal pack
            criterion = torch.nn.L1Loss()
            # loss_DiffRelation = criterion(Derivative_Data, y[:, 1:-1, DOF_Serie:])
            # loss_DiffRelation = myloss(Derivative_Data[:, :, :DOF_Serie], y[:, 1:-1, DOF_Serie: 2*DOF_Serie]) + myloss(Derivative_Data[:, :, DOF_Serie:], y[:, 1:-1, 2*DOF_Serie:])

            # loss_DiffRelation = myloss(Derivative_Data, y[:, 1:-1, DOF_flex:])

            loss_DiffRelation = torch.tensor([0]).to(torch.device('cuda:0'))

            total_loss = loss_f * f_weight + loss_f_virtual * fv_weight + data_loss * data_weight + loss_DiffRelation * diff_weight

            # total_loss = loss_f_virtual * fv_weight + data_loss * data_weight + loss_DiffRelation * diff_weight
            # total_loss = data_loss * data_weight + 1.0 * loss_DiffRelation

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            data_l2 += data_loss.item()
            train_pino += loss_f.item()
            virtual_pino += loss_f_virtual.item()
            train_loss += total_loss.item()
            DiffRelation_loss += loss_DiffRelation.item()
            Du_Magnitude = torch.zeros(1, 400)
            Magnitude_test += Du_Magnitude.detach().to(torch.device('cpu')).numpy()
            # Magnitude = (torch.max(Du_Magnitude) / torch.min(Du_Magnitude)).item()

        for x, y, PDE_weights in test_loader:
            x, y = x.to(ComDevice), y.to(ComDevice)
            # ManualSpy
            out = model(x).reshape(y.size(0), y.size(1), outputDim)
            data_loss = myloss(out, y[:, :, :DOF_flex]).detach().to(torch.device('cpu')).numpy()
            # Direct PDE loss of the signal
            loss_f, Du_Magnitude, Derivative_Data = BSA_PINO_loss(out, x, PDE_weights, ToOneV, inputDim, outputDim, D,
                                                                  DOF_exc, DOF_flex, W2_CX, W2_CY, W2_CZ, Eigens2,
                                                                  TrackDOFs,
                                                                  Nloc, ComDevice, switch=1)
            # loss_f = torch.tensor([0]).to(torch.device('cuda:0'))
            # Du_Magnitude = torch.tensor([0])
            # Diff Relation loss for the signal pack
            # Diff_Weight = torch.ones(20).to(torch.device('cuda:0'))
            # Diff_Weight[0], Diff_Weight[10] = 2, 2
            # loss_DiffRelation = myloss(Derivative_Data, y[:, 1:-1, DOF_Serie:])
            loss_DiffRelation = torch.tensor([0]).to(ComDevice)

            FirstDerivative_loss = myloss(Derivative_Data[:, :, :DOF_flex],
                                          y[:, 1:-1, DOF_flex:2 * DOF_flex]).detach().to(torch.device('cpu')).numpy()
            SecondDerivative_loss = myloss(Derivative_Data[:, :, DOF_flex:], y[:, 1:-1, 2 * DOF_flex:]).detach().to(
                torch.device('cpu')).numpy()
            total_loss = data_loss + FirstDerivative_loss + SecondDerivative_loss

            # total_loss = data_loss * data_weight
            train_pino_test += loss_f.item()
            train_loss_test += total_loss.item()
            DiffRelation_loss_test += loss_DiffRelation.item()
            SignalLoss_test += data_loss
            FirstDerivativeLoss_test += FirstDerivative_loss
            SecondDerivativeLoss_test += SecondDerivative_loss

        scheduler.step()
        data_l2 /= len(train_loader)
        train_pino /= len(train_loader)
        virtual_pino /= len(train_loader)
        train_loss /= len(train_loader)
        DiffRelation_loss /= len(train_loader)
        Magnitude_test /= len(train_loader)

        train_pino_test /= len(test_loader)
        train_loss_test /= len(test_loader)
        DiffRelation_loss_test /= len(test_loader)
        SignalLoss_test /= len(test_loader)
        FirstDerivativeLoss_test /= len(test_loader)
        SecondDerivativeLoss_test /= len(test_loader)
        # if use_tqdm:
        #     pbar.set_description(
        #         (
        #             f'Epoch {e}, train loss: {train_loss:.5f} '
        #             f'train f error: {train_pino * f_weight:.5f}; '
        #             f'data l2 error: {data_l2:.5f}; '
        #             f'Magnitude Situation: {Magnitude:.1f}'
        #         )
        #     )
        # if wandb and log:
        #     wandb.log(
        #         {
        #             'Train f error': train_pino,
        #             'Train L2 error': data_l2,
        #             'Train loss': train_loss,
        #         }
        #     )
        LossRecord[Index, 0:7] = [DiffRelation_loss, train_pino * f_weight, virtual_pino * fv_weight, data_l2,
                                  SignalLoss_test, FirstDerivativeLoss_test, SecondDerivativeLoss_test]
        LossRecord[Index, 7:] = Magnitude_test
        Index += 1
        if e % 250 == 0:
            save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                            model, optimizer)
        if e % 2 == 0:
            if PerformanceSwitch == 'On':
                np.savetxt('checkpoints/' + config['train']['save_dir'] + '/' + 'Performance' + str(Index) + '.txt',
                           torch.cat([out[0, :, :], y[0, :, :DOF_flex]], dim=-1).detach().to(
                               torch.device('cpu')).numpy())
            np.savetxt('checkpoints/' + config['train']['save_dir'] + '/' + config['train']['LossFileName'] + '.txt',
                       LossRecord)
        if Index >= 475:
            if np.sum(LossRecord[Index, -3:]) < np.sum(LossRecord[Index - 1, -3:]):
                save_checkpoint(config['train']['save_dir'],
                                config['train']['save_name'].replace('.pt', 'Best_On_Test.pt'),
                                model, optimizer)
        if Index >= 10:
            if e % 10 == 0:
                save_checkpoint(config['train']['save_dir'],
                                config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                                model, optimizer)
        print('train loss:{}; f loss:{}; fv loss:{}; DiffRelation loss:{}, data loss:{}'.format(train_loss,
                                                                                                train_pino * f_weight,
                                                                                                virtual_pino * fv_weight,
                                                                                                1.0 * DiffRelation_loss,
                                                                                                data_l2 * data_weight))
        print('test loss:{}; Signal loss:{}; FirstDerivative loss:{}, SecondDerivative loss:{}'.format(train_loss_test,
                                                                                                       SignalLoss_test,
                                                                                                       FirstDerivativeLoss_test,
                                                                                                       SecondDerivativeLoss_test))
        # print('Magnitude of PDE loss for different equations:', Magnitude_test)
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    print('Done!')
