import numpy as np
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import inspect


def FDD_FES(u, x, W2, Eigens2, TrackDOFs, Nloc, weight, ToOneV, inputDim, outputDim, D, DOF_exc, DOF_rigid, DOF_flex, ComDevice, DiffLossSwitch, g=9.8):
    batch_size = u.size(0)
    nt = u.size(1)
    nf = u.size(2)
    # DOF_exc = 3
    # DOF_rigid = 9
    # DOF_flex = (1 / 3) * (u.shape[-1] - 3 * DOF_rigid)

    # Extracting and ToOne actions for all parameters
    Excitation, U_flex, U_rigid, dU_flex, dU_rigid, ddU_flex, ddU_rigid = ToOne_Action_FES(ToOneV, x, u, D, DOF_exc, DOF_rigid, DOF_flex)

    # Coupling Points Information
    '''M is the mass, K is the spring stiffness, C is the damping, K_Contact is the contact stiffness. alpha_k and 
    alpha_c are the rayleigh damping coefficients.
    '''
    M = 1
    K = 1e4
    C = 1e2
    K_Contact = 1e4
    ConnectNum = 3
    alpha_k = 0.001
    alpha_c = 0.001

    ConnectDis = torch.matmul(U_flex, W2.transpose(0, 1))
    ConnectVel = torch.matmul(dU_flex, W2.transpose(0, 1))

    Fvec = torch.zeros(batch_size, nt-2, ConnectNum*3).to(ComDevice)

    # Encoding physics laws for Rigid systems
    Du_rigid = M * ddU_rigid + K * (U_rigid - ConnectDis) + C * (dU_rigid - ConnectVel)

    # Computing General Force for flexible modes
    Fvec[:, :, :ConnectNum] = ((K * (U_rigid - ConnectDis)) + (C * (dU_rigid - ConnectVel)))[:, :, :ConnectNum]
    Fvec[:, :, ConnectNum:2*ConnectNum] = ((K * (U_rigid - ConnectDis)) + (C * (dU_rigid - ConnectVel)))[:, :, ConnectNum:2*ConnectNum]
    Fvec[:, :, 2*ConnectNum:3*ConnectNum] = ((K * (U_rigid - ConnectDis)) + (C * (dU_rigid - ConnectVel)))[:, :, 2*ConnectNum:3*ConnectNum]
    Gvec = torch.matmul(Fvec, W2)

    # Encoding physics laws for Flexible modes
    Du_flex = ddU_flex - Gvec + Eigens2.squeeze(-1) * U_flex + (Eigens2.squeeze(-1) * alpha_k + alpha_c) * dU_flex

    # Computing excitation components in the equations
    Du_rigid[:, :, 2*ConnectNum:3*ConnectNum] -= K_Contact * (Excitation - U_rigid[:, :, 2*ConnectNum:3*ConnectNum])

    # Applying ODEs weights for all equations
    TargetMin = 0.025

    Du_flex *= TargetMin / weight.unsqueeze(-2)[:, :, :U_flex.size(-1)].to(ComDevice)
    Du_rigid *= TargetMin / weight.unsqueeze(-2)[:, :, U_flex.size(-1):].to(ComDevice)

    # Computing magnitude information for all DU
    DuStd = torch.cat((torch.std(Du_flex, [0, 1]), torch.std(Du_rigid, [0, 1])))
    DuMean = torch.cat((torch.mean(Du_flex, [0, 1]), torch.mean(Du_rigid, [0, 1])))
    Du_Magnitude = torch.cat((DuStd, DuMean))

    # Regenerate the data tensor for derivatives output
    # Optional action (Rely on DiffLossSwitch)
    if DiffLossSwitch == 'On':
        MeanV, StdV = ToOneV[0, 1:, 0], ToOneV[0, 1:, 3]
        DOF_Serie = DOF_flex + DOF_rigid
        DOF_de = DOF_exc + DOF_Serie
        dU_flex = (dU_flex - MeanV[DOF_de: DOF_de + DOF_flex]) / StdV[DOF_de: DOF_de + DOF_flex]
        dU_rigid = (dU_rigid - MeanV[DOF_de + DOF_flex: DOF_de + DOF_flex + DOF_rigid]) / StdV[DOF_de + DOF_flex: DOF_de + DOF_flex + DOF_rigid]
        DOF_dde = DOF_de + DOF_Serie
        ddU_flex = (ddU_flex - MeanV[DOF_dde: DOF_dde + DOF_flex]) / StdV[DOF_dde: DOF_dde + DOF_flex]
        ddU_rigid = (ddU_rigid - MeanV[DOF_dde + DOF_flex: DOF_dde + DOF_flex + DOF_rigid]) / StdV[DOF_dde + DOF_flex: DOF_dde + DOF_flex + DOF_rigid]

        Derivative_Data = torch.cat([dU_flex, dU_rigid, ddU_flex, ddU_rigid], dim=-1)
    else:
        Derivative_Data = torch.tensor([0]).to(ComDevice)
    return Du_flex, Du_rigid, Du_Magnitude, Derivative_Data


def FES_PINO_loss(u, x, weight, ToOneV, inputDim, outputDim, D, DOF_exc, DOF_rigid, DOF_flex, W2, Eigens2, TrackDOFs, Nloc, ComDevice, DiffLossSwitch):
    batchsize = u.size(0)
    nt = u.size(1)
    u = u.reshape(batchsize, nt, outputDim)

    Du_flex, Du_rigid, Du_Magnitude, Derivative_Data = FDD_FES(u, x, W2, Eigens2, TrackDOFs, Nloc, weight, ToOneV, inputDim, outputDim, D, DOF_exc, DOF_rigid, DOF_flex, ComDevice, DiffLossSwitch, g=9.8)
    # criterion = torch.nn.SmoothL1Loss()
    criterion = torch.nn.L1Loss()
    loss_f = criterion(Du_flex, torch.zeros_like(Du_flex)) + criterion(Du_rigid, torch.zeros_like(Du_rigid))
    return loss_f, Du_Magnitude, Derivative_Data


def Var_Encoder_FES(x, u, DOF_flex):
    U_flex = u[:, :, :DOF_flex]   # Output of flexible modes
    U_rigid = u[:, :, DOF_flex:]  # Output of rigid motions
    Excitation = x[:, 1:-1, 1:]   # Excitations
    return Excitation, U_flex, U_rigid


def ToOne_Action_FES(ToOneV, x, u, D, DOF_exc, DOF_rigid, DOF_flex):
    Excitation, U_flex, U_rigid = Var_Encoder_FES(x, u, DOF_flex)
    MeanV, StdV = ToOneV[0, 1:, 0], ToOneV[0, 1:, 3]
    # ReNorm for Excitations, Flexible modes and Rigid motions
    DOF_Serie = DOF_rigid + DOF_flex
    Excitation = Excitation * StdV[:DOF_exc] + MeanV[:DOF_exc]
    U_flex = U_flex * StdV[DOF_exc:DOF_exc + DOF_flex] + MeanV[DOF_exc:DOF_exc + DOF_flex]
    U_rigid = U_rigid * StdV[DOF_exc + DOF_flex:DOF_exc + DOF_Serie] + MeanV[DOF_exc + DOF_flex:DOF_exc + DOF_Serie]

    # Derivatives computation
    nt = u.size(1)
    dt = D / nt

    dU_rigid = (U_rigid[:, 2:, :] - U_rigid[:, :-2, :]) / (2 * dt)
    ddU_rigid = (U_rigid[:, 2:, :] - 2 * U_rigid[:, 1:-1, :] + U_rigid[:, :-2, :]) / (dt ** 2)
    dU_flex = (U_flex[:, 2:, :] - U_flex[:, :-2, :]) / (2 * dt)
    ddU_flex = (U_flex[:, 2:, :] - 2 * U_flex[:, 1:-1, :] + U_flex[:, :-2, :]) / (dt ** 2)

    # Reshape the solution of the system
    U_rigid = U_rigid[:, 1:-1, :]
    U_flex = U_flex[:, 1:-1, :]

    return Excitation, U_flex, U_rigid, dU_flex, dU_rigid, ddU_flex, ddU_rigid
