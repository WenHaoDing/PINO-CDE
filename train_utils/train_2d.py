import numpy as np
import torch
from tqdm import tqdm
from .utils import save_checkpoint
from .losses import LpLoss, darcy_loss, PINO_loss
from .losses_FES import FES_PINO_loss
from .losses_AMA import AMA_PINO_loss
from .losses_VTCD import VTCD_PINO_loss, VTCD_PINO_loss_Variant1
from .losses_BSA import BSA_PINO_loss
from .losses_HM import HM_PINO_loss
import os

try:
    import wandb
except ImportError:
    wandb = None


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
        DiffLossSwitch = config['data']['DiffLossSwitch']
        OperatorType = config['data']['OperatorType']
        Boundary = config['data']['Boundary']
        virtual_iter = iter(virtual_loader)
        NoData = config['data']['NoData']

        for x, y, PDE_weights in train_loader:
            x, y = x.to(ComDevice), y.to(ComDevice)
            DOF_exc = 3
            DOF_rigid = 9
            DOF_flex = int((1 / 3) * (y.shape[-1] - 3 * DOF_rigid))
            DOF_Serie = DOF_rigid + DOF_flex

            # ManualSpy
            # print('Shape of input is{}; output is{} and model output is{}'.format(x.shape, y.shape, model(x).shape))
            out = model(x).reshape(y.size(0), y.size(1), outputDim)
            if NoData == 'On':
                data_loss = torch.tensor([0]).to(ComDevice)
            else:
                data_loss = myloss(out, y[:, :, :DOF_Serie])  # Pure data losses
            # Direct PDE loss of the signal
            if OperatorType == 'PINO-MBD' or 'PINO':
                # For DiffLossSwitch='Off', Derivative_Data will automate be empty.
                loss_f, Du_Magnitude, Derivative_Data = FES_PINO_loss(out, x, PDE_weights, ToOneV, inputDim, outputDim,
                                                                      D,
                                                                      DOF_exc, DOF_rigid, DOF_flex, W2, Eigens2,
                                                                      TrackDOFs,
                                                                      Nloc, ComDevice, DiffLossSwitch)
                if DiffLossSwitch == 'On' and OperatorType != 'PINO':
                    if Boundary == 'On':
                        loss_DiffRelation = 0.5 * (myloss(Derivative_Data[:, :50, :], y[:, 1:(1+50), DOF_Serie:]) + myloss(Derivative_Data[:, -50:, :], y[:, -51:-1, DOF_Serie:]))
                    else:
                        loss_DiffRelation = myloss(Derivative_Data, y[:, 1:-1, DOF_Serie:])
                else:
                    loss_DiffRelation = torch.tensor([0]).to(ComDevice)
                if VirtualSwitch == 'On':
                    x_virtual = next(virtual_iter)[0].to(ComDevice)
                    out_virtual = model(x_virtual).reshape(x_virtual.size(0), y.size(1), outputDim)
                    virtual_weights = PDE_weights_virtual.double().repeat_interleave(x_virtual.size(0), dim=0)
                    loss_f_virtual, _, _ = FES_PINO_loss(out_virtual, x_virtual, virtual_weights, ToOneV, inputDim,
                                                         outputDim,
                                                         D, DOF_exc, DOF_rigid, DOF_flex, W2, Eigens2, TrackDOFs, Nloc,
                                                         ComDevice, 'Off')
                    # Virtual data is default to need no derivatives.
                else:
                    loss_f_virtual = torch.tensor([0]).to(ComDevice)

            else:
                loss_f = torch.tensor([0]).to(ComDevice)
                loss_DiffRelation = torch.tensor([0]).to(ComDevice)
                loss_f_virtual = torch.tensor([0]).to(ComDevice)
                Du_Magnitude = torch.zeros(1, 2 * (15 + 9)).to(ComDevice)

            # loss_f = torch.tensor([0]).to(torch.device('cuda:0'))
            # Du_Magnitude = torch.tensor([0])
            # Diff Relation loss for the signal pack
            # criterion = torch.nn.L1Loss()
            # loss_DiffRelation = criterion(Derivative_Data, y[:, 1:-1, DOF_Serie:])
            # loss_DiffRelation = myloss(Derivative_Data[:, :, :DOF_Serie], y[:, 1:-1, DOF_Serie: 2*DOF_Serie]) + myloss(Derivative_Data[:, :, DOF_Serie:], y[:, 1:-1, 2*DOF_Serie:])

            # loss_DiffRelation = torch.tensor([0]).to(torch.device('cuda:0'))
            # Total loss is default to have all 4 types of losses for unification.
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
                                                                  Nloc, ComDevice, 'On')
            # Test evaluation is default to have derivatives for inspection.
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
        if Index >= 475:
            if np.sum(LossRecord[Index, -3:]) < np.sum(LossRecord[Index - 1, -3:]):
                save_checkpoint(config['train']['save_dir'],
                                config['train']['save_name'].replace('.pt', 'Best_On_Test.pt'),
                                model, optimizer)
        if Index >= 50:
            if e % 100 == 0:
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


def train_VTCD_GradNorm(model,
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
    GradNorm_alpha = config['data']['GradNorm_alpha']

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
    VirtualSwitch = config['data']['VirtualSwitch']
    PerformanceSwitch = config['data']['PerformanceSwitch']
    DiffLossSwitch = config['data']['DiffLossSwitch']
    OperatorType = config['data']['OperatorType']
    NoData = config['data']['NoData']
    GradNorm = config['data']['GradNorm']
    BoundarySwitch = config['data']['Boundary']
    # GradNorm will turn off for single task scenario
    task_number = model.task_weights.size(0)
    if task_number == 1:
        GradNorm = 'Off'

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
        virtual_iter = iter(virtual_loader)


        for x, y, PDE_weights in train_loader:
            x, y = x.to(ComDevice), y.to(ComDevice)
            if NoData != 'On':
                out = model(x).reshape(y.size(0), y.size(1), outputDim)
                data_loss = myloss(out, torch.cat([y[:, :, :10], y[:, :, -4:]], dim=-1))
            if (OperatorType == 'PINO-MBD' or OperatorType == 'PINO') and NoData != 'On':
                loss_f, Du_Magnitude, Derivative_Data = VTCD_PINO_loss(out, x, PDE_weights, ToOneV, inputDim, outputDim, D, ComDevice)
                if DiffLossSwitch == 'On' and OperatorType != 'PINO':
                    if BoundarySwitch == 'On':
                        loss_DiffRelation = myloss(Derivative_Data[:, :50, :10], y[:, 1:1+50, 10:20]) + myloss(Derivative_Data[:, :50, 10:], y[:, 1:1+50, 20:-4]) \
                                            + myloss(Derivative_Data[:, -50:, :10], y[:, -51:-1, 10:20]) + myloss(Derivative_Data[:, -50:, 10:], y[:, -51:-1, 20:-4]) \
                                            + myloss(out[:, :50, :], torch.cat([y[:, :50, :10], y[:, :50, -4:]], dim=-1)) + myloss(out[:, -50:, :], torch.cat([y[:, -50:, :10], y[:, -50:, -4:]], dim=-1))
                    else:
                        loss_DiffRelation = myloss(Derivative_Data[:, :, :10], y[:, 1:-1, 10:20]) + myloss(Derivative_Data[:, :, 10:], y[:, 1:-1, 20:-4]) + \
                                            0.1 * (myloss(Derivative_Data[:, :50, :10], y[:, 1:1+50, 10:20]) + myloss(Derivative_Data[:, :50, 10:], y[:, 1:1+50, 20:-4]) +
                                                   myloss(Derivative_Data[:, -50:, :10], y[:, -51:-1, 10:20]) + myloss(Derivative_Data[:, -50:, 10:], y[:, -51:-1, 20:-4]) +
                                                   myloss(out[:, :50, :], torch.cat([y[:, :50, :10], y[:, :50, -4:]], dim=-1)) + myloss(out[:, -50:, :], torch.cat([y[:, -50:, :10], y[:, -50:, -4:]], dim=-1)))
                elif DiffLossSwitch == 'On' and OperatorType == 'PINO':
                    if BoundarySwitch == 'On':
                        loss_DiffRelation = myloss(Derivative_Data[:, :50, :10], y[:, 1:1+50, 10:20]) + myloss(Derivative_Data[:, :50, 10:], y[:, 1:1+50, 20:-4]) \
                                            + myloss(Derivative_Data[:, -50:, :10], y[:, -51:-1, 10:20]) + myloss(Derivative_Data[:, -50:, 10:], y[:, -51:-1, 20:-4]) \
                                            + myloss(out[:, :50, :], torch.cat([y[:, :50, :10], y[:, :50, -4:]], dim=-1)) + myloss(out[:, -50:, :], torch.cat([y[:, -50:, :10], y[:, -50:, -4:]], dim=-1))
                    else:
                        loss_DiffRelation = torch.tensor(0).to(ComDevice)
                else:
                    loss_DiffRelation = torch.tensor(0).to(ComDevice)
                if VirtualSwitch == 'On':
                    x_virtual = next(iter(virtual_iter))[0].to(ComDevice)
                    out_virtual = model(x_virtual).reshape(x_virtual.size(0), y.size(1), outputDim)
                    virtual_weights = PDE_weights_virtual.double().repeat_interleave(x_virtual.size(0), dim=0)
                    loss_f_virtual, _, _ = VTCD_PINO_loss(out_virtual, x_virtual, virtual_weights, ToOneV, inputDim,
                                                          outputDim, D, ComDevice)
                else:
                    loss_f_virtual = torch.tensor(0).to(ComDevice)
            elif NoData == 'On':
                x_virtual = next(iter(virtual_iter))[0].to(ComDevice)
                out_virtual = model(x_virtual).reshape(x_virtual.size(0), y.size(1), outputDim)
                virtual_weights = PDE_weights_virtual.double().repeat_interleave(x_virtual.size(0), dim=0)
                loss_f_virtual, _, _ = VTCD_PINO_loss(out_virtual, x_virtual, virtual_weights, ToOneV, inputDim,
                                                      outputDim, D, ComDevice)
                data_loss = torch.tensor([0]).to(ComDevice)
                loss_f = torch.tensor([0]).to(ComDevice)
                loss_DiffRelation = torch.tensor([0]).to(ComDevice)
                Du_Magnitude = torch.zeros(1, 2 * (10)).to(ComDevice)
            else:
                loss_f = torch.tensor([0]).to(ComDevice)
                loss_DiffRelation = torch.tensor([0]).to(ComDevice)
                loss_f_virtual = torch.tensor([0]).to(ComDevice)
                Du_Magnitude = torch.zeros(1, 2 * (10)).to(ComDevice)

            task_loss = []
            if OperatorType == 'PINO-MBD' or OperatorType == 'PINO':
                if NoData == 'On':
                    task_loss.append(loss_f_virtual)
                else:
                    task_loss.append(data_loss)
                    task_loss.append(loss_f)
                    if DiffLossSwitch == 'On':
                        task_loss.append(loss_DiffRelation)
                    if VirtualSwitch == 'On':
                        task_loss.append(loss_f_virtual)
            else:
                task_loss.append(data_loss)
            task_loss = torch.stack(task_loss)
            if Index == 0:
                initial_task_loss = task_loss.detach().cpu().numpy()
            if GradNorm == 'On':
                weighted_loss = torch.mul(model.task_weights, task_loss)
            else:
                weighted_loss = loss_f_virtual * fv_weight + data_loss * data_weight + loss_DiffRelation * diff_weight + loss_f * f_weight
            total_loss = torch.sum(weighted_loss)
            optimizer.zero_grad()
            if GradNorm == 'On':
                # backward with retain graph and zero the grad for weights
                total_loss.backward(retain_graph=True)
                model.task_weights.grad.data = model.task_weights.grad.data * 0.0
                # get the layer weights
                W = model.get_last_layer()
                # get the norms for each of the tasks
                norms = []
                # for i in range(len(initial_task_loss)):
                for i in range(len(task_loss)):
                    gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
                    norms.append(torch.norm(torch.mul(model.task_weights[i], gygw[0])))
                norms = torch.stack(norms)

                # compute the inverse training rate r_i(t)
                loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss
                inverse_train_rate = loss_ratio / np.mean(loss_ratio)
                # compute the mean norm
                mean_norm = np.mean(norms.data.cpu().numpy())

                # compute the GradNorm loss
                constant_term = torch.tensor(mean_norm * (inverse_train_rate ** GradNorm_alpha), requires_grad=False)
                constant_term = constant_term.cuda()
                grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
                model.task_weights.grad = torch.autograd.grad(grad_norm_loss, model.task_weights, allow_unused=True)[0]
            else:
                total_loss.backward()
                # model.task_weights.grad.data = model.task_weights.grad.data * 0.0
            # continue step with the optimizer
            optimizer.step()

            data_l2 += data_loss.item()
            train_pino += loss_f.item()
            virtual_pino += loss_f_virtual.item()
            train_loss += total_loss.item()
            DiffRelation_loss += loss_DiffRelation.item()
            Magnitude_test += Du_Magnitude.detach().to(torch.device('cpu')).numpy()
        # renormalize
        normalize_coeff = task_number / torch.sum(model.task_weights.data, dim=0)
        model.task_weights.data = model.task_weights.data * normalize_coeff
        print('Step at{}; task weights:{}'.format(Index, model.task_weights.data.detach().cpu().numpy()))

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
        LossRecord[Index, 0:7] = [DiffRelation_loss, train_pino, virtual_pino, data_l2,
                                  SignalLoss_test, FirstDerivativeLoss_test, SecondDerivativeLoss_test]
        LossRecord[Index, 7:] = Magnitude_test
        Index += 1
        if e % 50 == 0:
            save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                            model, optimizer)
        if e % 2 == 0:
            if PerformanceSwitch == 'On':
                np.savetxt('checkpoints/' + config['train']['save_dir'] + '/' + 'Performance' + str(Index) + '.txt',
                           torch.cat([out[0, :, :], torch.cat([y[0, :, :10], y[0, :, -4:]], dim=-1)],
                                     dim=-1).detach().to(
                               torch.device('cpu')).numpy())
            np.savetxt('checkpoints/' + config['train']['save_dir'] + '/' + config['train']['LossFileName'] + '.txt',
                       LossRecord)
        if Index >= 250:
            if np.sum(LossRecord[Index, -3:]) < np.sum(LossRecord[Index - 1, -3:]):
                save_checkpoint(config['train']['save_dir'],
                                config['train']['save_name'].replace('.pt', 'Best_On_Test.pt'),
                                model, optimizer)
        if Index == config['train']['epochs'] - 1:
            save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'].replace('.pt', 'Best_On_Test.pt'),
                            model, optimizer)
        # if Index >= 250:
        #     if e % 2 == 0:
        #         save_checkpoint(config['train']['save_dir'],
        #                         config['train']['save_name'].replace('.pt', f'_{e}.pt'),
        #                         model, optimizer)
        print('train loss:{}; f loss:{}; fv loss:{}; DiffRelation loss:{}, data loss:{}'.format(train_loss,
                                                                                                train_pino * f_weight,
                                                                                                virtual_pino * fv_weight,
                                                                                                DiffRelation_loss * diff_weight,
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


def train_BSA_WithGradNorm(model,
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
    GradNorm_alpha = config['data']['GradNorm_alpha']

    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    # Record of loss functions
    LossRecord = np.zeros((config['train']['epochs'] + 1, 7 + 200 * 2))
    Index = 0
    VirtualSwitch = config['data']['VirtualSwitch']
    PerformanceSwitch = config['data']['PerformanceSwitch']
    DiffLossSwitch = config['data']['DiffLossSwitch']
    OperatorType = config['data']['OperatorType']
    NoData = config['data']['NoData']
    GradNorm = config['data']['GradNorm']
    BoundarySwitch = config['data']['Boundary']
    # GradNorm will turn off for single task scenario
    task_number = model.task_weights.size(0)
    if task_number == 1:
        GradNorm = 'Off'

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
        virtual_iter = iter(virtual_loader)

        for x, y, PDE_weights in train_loader:
            x, y = x.to(ComDevice), y.to(ComDevice)
            DOF_exc = 6
            DOF_flex = int((1 / 3) * y.shape[-1])

            # ManualSpy
            # print('Shape of input is{}; output is{} and model output is{}'.format(x.shape, y.shape, model(x).shape))
            if NoData != 'On':
                out = model(x).reshape(y.size(0), y.size(1), outputDim)
                data_loss = myloss(out, y[:, :, :DOF_flex])  # Pure data losses
            # Direct PDE loss of the signal
            if (OperatorType == 'PINO-MBD' or OperatorType == 'PINO') and NoData != 'On':
                # For DiffLossSwitch='Off', Derivative_Data will automate be empty.
                loss_f, Du_Magnitude, Derivative_Data = BSA_PINO_loss(out, x, PDE_weights, ToOneV, inputDim, outputDim,
                                                                      D,
                                                                      DOF_exc, DOF_flex, W2_CX, W2_CY, W2_CZ, Eigens2,
                                                                      TrackDOFs,
                                                                      Nloc, ComDevice, switch=1)
                if DiffLossSwitch == 'On' and OperatorType != 'PINO':
                    if BoundarySwitch == 'On':
                        loss_DiffRelation = myloss(Derivative_Data[:, :25, :DOF_flex], y[:, 1:1+25, DOF_flex:2 * DOF_flex]) + myloss(Derivative_Data[:, :25, DOF_flex:], y[:, 1:1+25, 2 * DOF_flex:]) \
                                            + myloss(Derivative_Data[:, -25:, :DOF_flex], y[:, -26:-1, DOF_flex:2 * DOF_flex]) + myloss(Derivative_Data[:, -25:, DOF_flex:], y[:, -26:-1, 2 * DOF_flex:]) + \
                                            myloss(out[:, :25, :], y[:, :25, :DOF_flex]) + myloss(out[:, -25:, :], y[:, -25:, :DOF_flex])
                    else:
                        loss_DiffRelation = myloss(Derivative_Data[:, :, :DOF_flex], y[:, 1:-1, DOF_flex:2 * DOF_flex]) + myloss(Derivative_Data[:, :, DOF_flex:], y[:, 1:-1, 2 * DOF_flex:]) +\
                                            0.01 * (myloss(Derivative_Data[:, :25, :DOF_flex], y[:, 1:1+25, DOF_flex:2 * DOF_flex]) + myloss(Derivative_Data[:, :25, DOF_flex:], y[:, 1:1+25, 2 * DOF_flex:]) +
                                                    myloss(Derivative_Data[:, -25:, :DOF_flex], y[:, -26:-1, DOF_flex:2 * DOF_flex]) + myloss(Derivative_Data[:, -25:, DOF_flex:], y[:, -26:-1, 2 * DOF_flex:]) +
                                                    myloss(out[:, :25, :], y[:, :25, :DOF_flex]) + myloss(out[:, -25:, :], y[:, -25:, :DOF_flex]))

                elif DiffLossSwitch == 'On' and OperatorType == 'PINO':
                    if BoundarySwitch == 'On':
                        loss_DiffRelation = myloss(Derivative_Data[:, :10, :DOF_flex], y[:, 1:1+10, DOF_flex:2 * DOF_flex]) + myloss(Derivative_Data[:, :10, DOF_flex:], y[:, 1:1+10, 2 * DOF_flex:]) \
                                            + myloss(Derivative_Data[:, -10:, :DOF_flex], y[:, -11:-1, DOF_flex:2 * DOF_flex]) + myloss(Derivative_Data[:, -10:, DOF_flex:], y[:, -11:-1, 2 * DOF_flex:]) + \
                                            myloss(out[:, :25, :], y[:, :25, :DOF_flex]) + myloss(out[:, -25:, :], y[:, -25:, :DOF_flex])
                    else:
                        loss_DiffRelation = torch.tensor(0).to(ComDevice)
                if VirtualSwitch == 'On':
                    x_virtual = next(iter(virtual_loader))[0].to(ComDevice)
                    out_virtual = model(x_virtual).reshape(x_virtual.size(0), y.size(1), outputDim)
                    virtual_weights = PDE_weights_virtual.double().repeat_interleave(x_virtual.size(0), dim=0)
                    loss_f_virtual, _, _ = BSA_PINO_loss(out_virtual, x_virtual, virtual_weights, ToOneV, inputDim,
                                                         outputDim, D, DOF_exc, DOF_flex, W2_CX, W2_CY, W2_CZ, Eigens2,
                                                         TrackDOFs,
                                                         Nloc, ComDevice, switch=0)
                    # Virtual data is default to need no derivatives.
                else:
                    loss_f_virtual = torch.tensor(0).to(ComDevice)

            elif NoData == 'On':
                x_virtual = next(virtual_iter)[0].to(ComDevice)
                out_virtual = model(x_virtual).reshape(x_virtual.size(0), y.size(1), outputDim)
                virtual_weights = PDE_weights_virtual.double().repeat_interleave(x_virtual.size(0), dim=0)
                loss_f_virtual, _, _ = BSA_PINO_loss(out_virtual, x_virtual, virtual_weights, ToOneV, inputDim,
                                                     outputDim, D, DOF_exc, DOF_flex, W2_CX, W2_CY, W2_CZ, Eigens2,
                                                     TrackDOFs,
                                                     Nloc, ComDevice, switch=0)
                data_loss = torch.tensor([0]).to(ComDevice)
                loss_f = torch.tensor([0]).to(ComDevice)
                loss_DiffRelation = torch.tensor([0]).to(ComDevice)
                Du_Magnitude = torch.zeros(1, 2 * (200)).to(ComDevice)
            else:
                loss_f = torch.tensor([0]).to(ComDevice)
                loss_DiffRelation = torch.tensor([0]).to(ComDevice)
                loss_f_virtual = torch.tensor([0]).to(ComDevice)
                Du_Magnitude = torch.zeros(1, 2 * (200)).to(ComDevice)

            task_loss = []
            if OperatorType == 'PINO-MBD' or OperatorType == 'PINO':
                if NoData == 'On':
                    task_loss.append(loss_f_virtual)
                else:
                    task_loss.append(data_loss)
                    task_loss.append(loss_f)
                    if DiffLossSwitch == 'On':
                        task_loss.append(loss_DiffRelation)
                    if VirtualSwitch == 'On':
                        task_loss.append(loss_f_virtual)
            else:
                task_loss.append(data_loss)
            task_loss = torch.stack(task_loss)
            if Index == 0:
                initial_task_loss = task_loss.detach().cpu().numpy()
            if GradNorm == 'On':
                weighted_loss = torch.mul(model.task_weights, task_loss)
            else:
                weighted_loss = loss_f_virtual * fv_weight + data_loss * data_weight + loss_DiffRelation * diff_weight + loss_f * f_weight
            total_loss = torch.sum(weighted_loss)
            optimizer.zero_grad()

            # Apply GradNorm for different losses
            if GradNorm == 'On':
                # backward with retain graph and zero the grad for weights
                total_loss.backward(retain_graph=True)
                model.task_weights.grad.data = model.task_weights.grad.data * 0.0
                # get the layer weights
                W = model.get_last_layer()
                # get the norms for each of the tasks
                norms = []
                # for i in range(len(initial_task_loss)):
                for i in range(len(task_loss)):
                    gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
                    norms.append(torch.norm(torch.mul(model.task_weights[i], gygw[0])))
                norms = torch.stack(norms)

                # compute the inverse training rate r_i(t)
                loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss
                inverse_train_rate = loss_ratio / np.mean(loss_ratio)
                # compute the mean norm
                mean_norm = np.mean(norms.data.cpu().numpy())

                # compute the GradNorm loss
                constant_term = torch.tensor(mean_norm * (inverse_train_rate ** GradNorm_alpha), requires_grad=False)
                constant_term = constant_term.cuda()
                grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
                model.task_weights.grad = torch.autograd.grad(grad_norm_loss, model.task_weights, allow_unused=True)[0]
            else:
                total_loss.backward()
                # model.task_weights.grad.data = model.task_weights.grad.data * 0.0
            # continue step with the optimizer
            optimizer.step()

            data_l2 += data_loss.item()
            train_pino += loss_f.item()
            virtual_pino += loss_f_virtual.item()
            train_loss += total_loss.item()
            DiffRelation_loss += loss_DiffRelation.item()
            Magnitude_test += Du_Magnitude.detach().to(torch.device('cpu')).numpy()
            # Magnitude = (torch.max(Du_Magnitude) / torch.min(Du_Magnitude)).item()
            # renormalize
        normalize_coeff = task_number / torch.sum(model.task_weights.data, dim=0)
        model.task_weights.data = model.task_weights.data * normalize_coeff
        print('Step at{}; task weights:{}'.format(Index, model.task_weights.data.detach().cpu().numpy()))

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
        LossRecord[Index, 0:7] = [DiffRelation_loss, train_pino, virtual_pino, data_l2,
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
                np.savetxt(
                    'checkpoints/' + config['train']['save_dir'] + '/' + config['train']['LossFileName'] + '.txt',
                    LossRecord)
        if Index >= 300:
            if e % 20 == 0:
                save_checkpoint(config['train']['save_dir'],
                                config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                                model, optimizer)
                np.savetxt(
                    'checkpoints/' + config['train']['save_dir'] + '/' + config['train']['LossFileName'] + '.txt',
                    LossRecord)
        print('train loss:{}; f loss:{}; fv loss:{}; DiffRelation loss:{}, data loss:{}'.format(train_loss,
                                                                                                train_pino,
                                                                                                virtual_pino,
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


def train_HM_GradNorm(model,
                      train_loader, test_loader, virtual_loader, PDE_weights_virtual,
                      optimizer, scheduler,
                      config,
                      ToOneV,
                      W2_1, W2_2, W2_3, Eigens2_1, Eigens2_2,
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

    # data_weight = torch.Tensor(np.array(config['train']['xy_loss']))
    # f_weight = torch.Tensor(np.array(config['train']['f_loss']))
    # fv_weight = torch.Tensor(np.array(config['train']['fv_loss']))
    # diff_weight = torch.Tensor(np.array(config['train']['diff_loss']))
    # task_weights = torch.stack((data_weight, f_weight, fv_weight, diff_weight))
    # task_weights = torch.stack((data_weight, f_weight, diff_weight))
    # task_weights.requires_grad = True
    GradNorm_alpha = config['data']['GradNorm_alpha']

    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    # Record of loss functions
    LossRecord = np.zeros((config['train']['epochs'] + 1, 7 + (15 + 10 + 9 + 9) * 2))
    Index = 0
    VirtualSwitch = config['data']['VirtualSwitch']
    PerformanceSwitch = config['data']['PerformanceSwitch']
    DiffLossSwitch = config['data']['DiffLossSwitch']
    OperatorType = config['data']['OperatorType']
    NoData = config['data']['NoData']
    GradNorm = config['data']['GradNorm']
    # GradNorm will turn off for single task scenario
    task_number = model.task_weights.size(0)
    if task_number == 1:
        GradNorm = 'Off'

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
        Magnitude_test = np.zeros((1, 2 * (15 + 10 + 9 + 9)))
        DiffRelation_loss_test = 0.0
        SignalLoss_test = 0.0
        FirstDerivativeLoss_test = 0.0
        SecondDerivativeLoss_test = 0.0
        virtual_iter = iter(virtual_loader)

        for x, y, PDE_weights in train_loader:
            x, y = x.to(ComDevice), y.to(ComDevice)
            DOF_exc = 3
            # DOF_rigid = 9
            # DOF_flex = int((1 / 3) * (y.shape[-1] - 3 * DOF_rigid))
            DOF_Serie = 15 + 10 + 9 * 2

            # ManualSpy
            # print('Shape of input is{}; output is{} and model output is{}'.format(x.shape, y.shape, model(x).shape))
            if NoData != 'On':
                out = model(x).reshape(y.size(0), y.size(1), outputDim)
                # data_loss = myloss(out, y[:, :, :DOF_Serie])  # Pure data losses
                data_loss = myloss(out[:, :100, :], y[:, :100, :DOF_Serie])
            # Direct PDE loss of the signal
            if (OperatorType == 'PINO-MBD' or OperatorType == 'PINO') and NoData != 'On':
                # For DiffLossSwitch='Off', Derivative_Data will automate be empty.
                loss_f, Du_Magnitude, Derivative_Data = HM_PINO_loss(out, x, PDE_weights, ToOneV, D, W2_1, W2_2, W2_3,
                                                                     Eigens2_1, Eigens2_2, ComDevice, DiffLossSwitch)
                if DiffLossSwitch == 'On' and OperatorType != 'PINO':
                    loss_DiffRelation = myloss(Derivative_Data, y[:, 1:-1, DOF_Serie:])
                else:
                    loss_DiffRelation = torch.tensor([0]).to(ComDevice)
                if VirtualSwitch == 'On':
                    x_virtual = next(virtual_iter)[0].to(ComDevice)
                    out_virtual = model(x_virtual).reshape(x_virtual.size(0), y.size(1), outputDim)
                    virtual_weights = PDE_weights_virtual.double().repeat_interleave(x_virtual.size(0), dim=0)
                    loss_f_virtual, _, _ = HM_PINO_loss(out_virtual, x_virtual, virtual_weights, ToOneV, D, W2_1, W2_2,
                                                        W2_3, Eigens2_1, Eigens2_2, ComDevice, 'Off')
                    # Virtual data is default to need no derivatives.
                else:
                    loss_f_virtual = torch.tensor(0).to(ComDevice)

            elif NoData == 'On':
                x_virtual = next(virtual_iter)[0].to(ComDevice)
                out_virtual = model(x_virtual).reshape(x_virtual.size(0), y.size(1), outputDim)
                virtual_weights = PDE_weights_virtual.double().repeat_interleave(x_virtual.size(0), dim=0)
                loss_f_virtual, _, _ = HM_PINO_loss(out_virtual, x_virtual, virtual_weights, ToOneV, D, W2_1, W2_2,
                                                    W2_3, Eigens2_1, Eigens2_2, ComDevice, 'Off')
                data_loss = torch.tensor([0]).to(ComDevice)
                # out = model(x).reshape(y.size(0), y.size(1), outputDim)
                # data_loss = myloss(out[:, :100, :], y[:, :100, :DOF_Serie])
                loss_f = torch.tensor([0]).to(ComDevice)
                loss_DiffRelation = torch.tensor([0]).to(ComDevice)
                Du_Magnitude = torch.zeros(1, 20).to(ComDevice)
            else:
                loss_f = torch.tensor([0]).to(ComDevice)
                loss_DiffRelation = torch.tensor([0]).to(ComDevice)
                loss_f_virtual = torch.tensor([0]).to(ComDevice)
                Du_Magnitude = torch.zeros(1, 20).to(ComDevice)

            # loss_f = torch.tensor([0]).to(torch.device('cuda:0'))
            # Du_Magnitude = torch.tensor([0])
            # Diff Relation loss for the signal pack
            # criterion = torch.nn.L1Loss()
            # loss_DiffRelation = criterion(Derivative_Data, y[:, 1:-1, DOF_Serie:])
            # loss_DiffRelation = myloss(Derivative_Data[:, :, :DOF_Serie], y[:, 1:-1, DOF_Serie: 2*DOF_Serie]) + myloss(Derivative_Data[:, :, DOF_Serie:], y[:, 1:-1, 2*DOF_Serie:])

            # loss_DiffRelation = torch.tensor([0]).to(torch.device('cuda:0'))
            # Total loss is default to have all 4 types of losses for unification.

            # total_loss = loss_f_virtual * fv_weight + data_loss * data_weight + loss_DiffRelation * diff_weight
            # Recording initial loss L(0) for the first step
            # task_loss = torch.stack((data_loss, loss_f, loss_f_virtual, loss_DiffRelation))

            # total_loss = data_loss * data_weight + loss_f * f_weight + loss_f_virtual * fv_weight + loss_DiffRelation * diff_weight
            task_loss = []
            if OperatorType == 'PINO-MBD' or OperatorType == 'PINO':
                if NoData == 'On':
                    task_loss.append(loss_f_virtual)
                else:
                    task_loss.append(data_loss)
                    task_loss.append(loss_f)
                    if DiffLossSwitch == 'On':
                        task_loss.append(loss_DiffRelation)
                    if VirtualSwitch == 'On':
                        task_loss.append(loss_f_virtual)
            else:
                task_loss.append(data_loss)
            task_loss = torch.stack(task_loss)
            if Index == 0:
                initial_task_loss = task_loss.detach().cpu().numpy()
            weighted_loss = torch.mul(model.task_weights, task_loss)
            total_loss = torch.sum(weighted_loss)
            optimizer.zero_grad()

            # Apply GradNorm for different losses
            if GradNorm == 'On':
                # backward with retain graph and zero the grad for weights
                total_loss.backward(retain_graph=True)
                model.task_weights.grad.data = model.task_weights.grad.data * 0.0
                # get the layer weights
                W = model.get_last_layer()
                # get the norms for each of the tasks
                norms = []
                # for i in range(len(initial_task_loss)):
                for i in range(len(task_loss)):
                    gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
                    norms.append(torch.norm(torch.mul(model.task_weights[i], gygw[0])))
                norms = torch.stack(norms)

                # compute the inverse training rate r_i(t)
                loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss
                inverse_train_rate = loss_ratio / np.mean(loss_ratio)
                # compute the mean norm
                mean_norm = np.mean(norms.data.cpu().numpy())

                # compute the GradNorm loss
                constant_term = torch.tensor(mean_norm * (inverse_train_rate ** GradNorm_alpha), requires_grad=False)
                constant_term = constant_term.cuda()
                grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
                model.task_weights.grad = torch.autograd.grad(grad_norm_loss, model.task_weights, allow_unused=True)[0]
            else:
                total_loss.backward()
                model.task_weights.grad.data = model.task_weights.grad.data * 0.0
            # continue step with the optimizer
            optimizer.step()

            data_l2 += data_loss.item()
            train_pino += loss_f.item()
            virtual_pino += loss_f_virtual.item()
            train_loss += total_loss.item()
            DiffRelation_loss += loss_DiffRelation.item()
            Magnitude_test += Du_Magnitude.detach().to(torch.device('cpu')).numpy()
            # Magnitude = (torch.max(Du_Magnitude) / torch.min(Du_Magnitude)).item()
        # renormalize
        normalize_coeff = task_number / torch.sum(model.task_weights.data, dim=0)
        model.task_weights.data = model.task_weights.data * normalize_coeff
        print('Step at{}; task weights:{}'.format(Index, model.task_weights.data.detach().cpu().numpy()))

        for x, y, PDE_weights in test_loader:
            x, y = x.to(ComDevice), y.to(ComDevice)
            # ManualSpy
            out = model(x).reshape(y.size(0), y.size(1), outputDim)
            data_loss = myloss(out, y[:, :, :DOF_Serie]).detach().to(torch.device('cpu')).numpy()
            # Direct PDE loss of the signal
            loss_f, Du_Magnitude, Derivative_Data = HM_PINO_loss(out, x, PDE_weights, ToOneV, D, W2_1, W2_2, W2_3,
                                                                 Eigens2_1, Eigens2_2, ComDevice, 'On')

            # Test evaluation is default to have derivatives for inspection.
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
        LossRecord[Index, 0:7] = [DiffRelation_loss, train_pino, virtual_pino, data_l2,
                                  SignalLoss_test, FirstDerivativeLoss_test, SecondDerivativeLoss_test]
        LossRecord[Index, 7:] = Magnitude_test
        Index += 1
        # if e % 250 == 0:
        #     save_checkpoint(config['train']['save_dir'],
        #                     config['train']['save_name'].replace('.pt', f'_{e}.pt'),
        #                     model, optimizer)
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
        if Index >= 475:
            if np.sum(LossRecord[Index, -3:]) < np.sum(LossRecord[Index - 1, -3:]):
                save_checkpoint(config['train']['save_dir'],
                                config['train']['save_name'].replace('.pt', 'Best_On_Test.pt'),
                                model, optimizer)
        if Index >= 450:
            if e % 100 == 0:
                save_checkpoint(config['train']['save_dir'],
                                config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                                model, optimizer)
        print('train loss:{}; f loss:{}; fv loss:{}; DiffRelation loss:{}, data loss:{}'.format(train_loss,
                                                                                                train_pino,
                                                                                                virtual_pino,
                                                                                                DiffRelation_loss,
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

