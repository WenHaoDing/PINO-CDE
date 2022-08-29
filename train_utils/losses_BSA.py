import numpy as np
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import inspect


def FDD_BSA(U_flex, x, W2_CX, W2_CY, W2_CZ, Eigens2, TrackDOFs, Nloc, weight, ToOneV, inputDim, outputDim, D, DOF_exc, DOF_flex, ComDevice, switch, g=9.8):
    batch_size = U_flex.size(0)
    nt = U_flex.size(1)
    nf = U_flex.size(2)
    # DOF_exc = 3
    # DOF_rigid = 9
    # DOF_flex = (1 / 3) * (u.shape[-1] - 3 * DOF_rigid)

    # Extracting and ToOne actions for all parameters
    Excitation, U_flex, dU_flex, ddU_flex = ToOne_Action_BSA(ToOneV, x, U_flex, D, DOF_exc, DOF_flex)

    # for i in range(0, ddU_rigid.size(-1)):
    #     plt.figure(i)
    #     plt.plot(ddU_rigid[0, :, i].to(torch.device('cpu')).numpy())
    #     plt.show()
    # Coupling Points Information
    K = 5e6
    C = 1e6
    ConnectNum = 3
    alpha_k = 0.001
    alpha_c = 0.001
    nDOF = TrackDOFs.size(0)
    TrackDOFs = TrackDOFs.long()


    # Computing General Mode Forces for flexible system (Compute the connection displacement and velocity first)
    ConnectDis = torch.cat((Excitation[:, :, 0].unsqueeze(-1).repeat(1, 1, nDOF) - torch.matmul(U_flex, W2_CX.transpose(0, 1)),
                            Excitation[:, :, 1].unsqueeze(-1).repeat(1, 1, nDOF) - torch.matmul(U_flex, W2_CY.transpose(0, 1)),
                            Excitation[:, :, 2].unsqueeze(-1).repeat(1, 1, nDOF) - torch.matmul(U_flex, W2_CZ.transpose(0, 1))), dim=-1)
    ConnectVel = torch.cat((Excitation[:, :, 3].unsqueeze(-1).repeat(1, 1, nDOF) - torch.matmul(dU_flex, W2_CX.transpose(0, 1)),
                            Excitation[:, :, 4].unsqueeze(-1).repeat(1, 1, nDOF) - torch.matmul(dU_flex, W2_CY.transpose(0, 1)),
                            Excitation[:, :, 5].unsqueeze(-1).repeat(1, 1, nDOF) - torch.matmul(dU_flex, W2_CZ.transpose(0, 1))), dim=-1)
    # print('The connection information dimension is {}'.format(ConnectDis.shape))

    Gvec = torch.matmul(K * ConnectDis + C * ConnectVel,
                        torch.cat((W2_CX, W2_CY, W2_CZ), dim=0))

    # Encoding physics laws for Flexible modes
    Du_flex = ddU_flex - Gvec + Eigens2.squeeze(-1) * U_flex + (Eigens2.squeeze(-1) * alpha_k + alpha_c) * dU_flex

    # Applying ODEs weights for all equations, the variable TargetMin is equal to r in Algorithm 1.
    TargetMin = 0.02

    Du_flex *= TargetMin / weight.unsqueeze(-2)[:, :, :U_flex.size(-1)].to(ComDevice)

    # Computing magnitude information for all DU
    Du_Magnitude = torch.cat((torch.std(Du_flex, [0, 1]), torch.mean(Du_flex, [0, 1])))

    # Regenerate the data tensor for derivatives output
    MeanV, StdV = ToOneV[0, 1:, 0], ToOneV[0, 1:, 3]
    DOF_de = DOF_exc + DOF_flex
    dU_flex = (dU_flex - MeanV[DOF_de: DOF_de + DOF_flex]) / StdV[DOF_de: DOF_de + DOF_flex]
    DOF_dde = DOF_de + DOF_flex
    ddU_flex = (ddU_flex - MeanV[DOF_dde: DOF_dde + DOF_flex]) / StdV[DOF_dde: DOF_dde + DOF_flex]

    if switch == 1:
        Derivative_Data = torch.cat([dU_flex, ddU_flex], dim=-1)
    else:
        Derivative_Data = torch.zeros(1, 1).to(ComDevice)
    return Du_flex, Du_Magnitude, Derivative_Data


def BSA_PINO_loss(U_flex, x, weight, ToOneV, inputDim, outputDim, D, DOF_exc, DOF_flex, W2_CX, W2_CY, W2_CZ, Eigens2, TrackDOFs, Nloc, ComDevice, switch):
    batchsize = U_flex.size(0)
    nt = U_flex.size(1)
    U_flex = U_flex.reshape(batchsize, nt, outputDim)

    Du_flex, Du_Magnitude, Derivative_Data = FDD_BSA(U_flex, x, W2_CX, W2_CY, W2_CZ, Eigens2, TrackDOFs, Nloc, weight, ToOneV, inputDim, outputDim, D, DOF_exc, DOF_flex, ComDevice, switch, g=9.8)
    # criterion = torch.nn.SmoothL1Loss()
    criterion = torch.nn.L1Loss()
    loss_f = criterion(Du_flex, torch.zeros_like(Du_flex))
    return loss_f, Du_Magnitude, Derivative_Data


def Var_Encoder_BSA(x):
    Excitation = x[:, 1:-1, 1:]   # Excitations
    return Excitation


def ToOne_Action_BSA(ToOneV, x, U_flex, D, DOF_exc, DOF_flex):
    Excitation = Var_Encoder_BSA(x)
    MeanV, StdV = ToOneV[0, 1:, 0], ToOneV[0, 1:, 3]
    # ReNorm for Excitations, Flexible modes and Rigid motions
    Excitation = Excitation * StdV[:DOF_exc] + MeanV[:DOF_exc]
    U_flex = U_flex * StdV[DOF_exc:DOF_exc + DOF_flex] + MeanV[DOF_exc:DOF_exc + DOF_flex]

    # Derivatives computation
    nt = U_flex.size(1)
    dt = D / nt

    dU_flex = (U_flex[:, 2:, :] - U_flex[:, :-2, :]) / (2 * dt)
    ddU_flex = (U_flex[:, 2:, :] - 2 * U_flex[:, 1:-1, :] + U_flex[:, :-2, :]) / (dt ** 2)

    # Reshape the solution of the system
    U_flex = U_flex[:, 1:-1, :]

    return Excitation, U_flex, dU_flex, ddU_flex
