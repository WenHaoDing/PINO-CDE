import numpy as np
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import inspect
import scipy.io as io


def FDD_HM(u, x, W2_1, W2_2, W2_3, Eigens2_1, Eigens2_2, weight, ToOneV, D, ComDevice, DiffLossSwitch):
    batch_size = u.size(0)
    nt = u.size(1)
    nf = u.size(2)
    # DOF_exc = 3
    # DOF_rigid = 9
    # DOF_flex = (1 / 3) * (u.shape[-1] - 3 * DOF_rigid)

    # Extracting and ToOne actions for all parameters
    Excitation, K1, K2, C1, C2, M1, M2, U_flex1, U_flex2, U_rigid1, U_rigid2, dU_flex1, dU_flex2, dU_rigid1, \
    dU_rigid2, ddU_flex1, ddU_flex2, ddU_rigid1, ddU_rigid2 = ToOne_Action_HM(ToOneV, x, u, D)

    # Coupling Points Information
    K_Contact = 1e4
    ConnectNum1 = 6
    ConnectNum2 = 3
    alpha_k = 0.001
    alpha_c = 0.001
    ConnectDis_p1 = torch.matmul(U_flex1, W2_1.transpose(0, 1))
    ConnectVel_p1 = torch.matmul(dU_flex1, W2_1.transpose(0, 1))
    ConnectDis_p2 = torch.matmul(U_flex1, W2_2.transpose(0, 1))
    ConnectVel_p2 = torch.matmul(dU_flex1, W2_2.transpose(0, 1))
    ConnectDis_p3 = torch.matmul(U_flex2, W2_3.transpose(0, 1))
    ConnectVel_p3 = torch.matmul(dU_flex2, W2_3.transpose(0, 1))
    # Fvec1 = torch.zeros(batch_size, nt - 2, 3 * ConnectNum1).to(ComDevice)
    # Fvec2 = torch.zeros(batch_size, nt - 2, 3 * ConnectNum2).to(ComDevice)
    # Embed physics laws for Rigid systems
    Du_rigid1 = M1 * ddU_rigid1 + K1 * (U_rigid1 - ConnectDis_p1) + C1 * (dU_rigid1 - ConnectVel_p1)
    Du_rigid2 = M2 * ddU_rigid2 + K2 * (2 * U_rigid2 - ConnectDis_p2 - ConnectDis_p3) + C2 * (2 * dU_rigid2 - ConnectVel_p2 - ConnectVel_p3)
    # Computing General Force for flexible modes
    Fvec1 = torch.cat((K1 * (U_rigid1 - ConnectDis_p1) + C1 * (dU_rigid1 - ConnectVel_p1), K2 * (U_rigid2 - ConnectDis_p2) + C2 * (dU_rigid2 - ConnectVel_p2)), dim=-1)
    Gvec1 = torch.matmul(Fvec1, torch.cat((W2_1, W2_2), dim=0))
    Fvec2 = K2 * (U_rigid2 - ConnectDis_p3) + C2 * (dU_rigid2 - ConnectVel_p3)
    Gvec2 = torch.matmul(Fvec2, W2_3)
    # plt.figure(1)
    # plt.plot(Gvec1[0, :, :].to(torch.device('cpu')).numpy())
    # plt.figure(2)
    # plt.plot(Gvec2[0, :, :].to(torch.device('cpu')).numpy())
    # plt.show()
    # Encoding physics laws for Flexible modes
    Du_flex1 = ddU_flex1 - Gvec1 + Eigens2_1.squeeze(-1) * U_flex1 + (Eigens2_1.squeeze(-1) * alpha_k + alpha_c) * dU_flex1
    Du_flex2 = ddU_flex2 - Gvec2 + Eigens2_2.squeeze(-1) * U_flex2 + (Eigens2_2.squeeze(-1) * alpha_k + alpha_c) * dU_flex2
    # Computing excitation components in the equations
    Du_rigid1[:, :, 6:9] -= K_Contact * (Excitation - U_rigid1[:, :, 6:9])
    # Applying ODEs weights for all equations
    TargetMin = 0.025
    Du_flex1 *= TargetMin / weight.unsqueeze(-2)[:, :, :15].to(ComDevice)
    Du_flex2 *= TargetMin / weight.unsqueeze(-2)[:, :, 15:25].to(ComDevice)
    Du_rigid1 *= TargetMin / weight.unsqueeze(-2)[:, :, 25:34].to(ComDevice)
    Du_rigid2 *= TargetMin / weight.unsqueeze(-2)[:, :, 34:43].to(ComDevice)

    DuStd = torch.cat((torch.std(Du_flex1, [0, 1]), torch.std(Du_flex2, [0, 1]), torch.std(Du_rigid1, [0, 1]), torch.std(Du_rigid2, [0, 1])))
    DuMean = torch.cat((torch.mean(Du_flex1, [0, 1]), torch.mean(Du_flex2, [0, 1]), torch.mean(Du_rigid1, [0, 1]), torch.mean(Du_rigid2, [0, 1])))
    Du_Magnitude = torch.cat((DuStd, DuMean))

    # Regenerate the data tensor for derivatives output
    # Optional action (Rely on DiffLossSwitch)
    if DiffLossSwitch == 'On':
        MeanV_out, StdV_out = ToOneV[0, (10 + 15 + 10 + 18):, 0], ToOneV[0, (10 + 15 + 10 + 18):, 3]
        MeanVd, StdVd, MeanVdd, StdVdd = MeanV_out[:43], StdV_out[:43], MeanV_out[43:], StdV_out[43:]
        dU_flex1 = (dU_flex1 - MeanVd[:15]) / StdVd[:15]
        ddU_flex1 = (ddU_flex1 - MeanVdd[:15]) / StdVdd[:15]
        dU_flex2 = (dU_flex2 - MeanVd[15:(15+10)]) / StdVd[15:(15+10)]
        ddU_flex2 = (ddU_flex2 - MeanVdd[15:(15+10)]) / StdVdd[15:(15+10)]
        dU_rigid1 = (dU_rigid1 - MeanVd[25:34]) / StdVd[25:34]
        ddU_rigid1 = (ddU_rigid1 - MeanVdd[25:34]) / StdVdd[25:34]
        dU_rigid2 = (dU_rigid2 - MeanVd[34:]) / StdVd[34:]
        ddU_rigid2 = (ddU_rigid2 - MeanVdd[34:]) / StdVdd[34:]
        Derivative_Data = torch.cat([dU_flex1, dU_flex2, dU_rigid1, dU_rigid2, ddU_flex1, ddU_flex2, ddU_rigid1, ddU_rigid2], dim=-1)
    else:
        Derivative_Data = torch.tensor([0]).to(ComDevice)
    return Du_flex1, Du_flex2, Du_rigid1, Du_rigid2, Du_Magnitude, Derivative_Data


def HM_PINO_loss(u, x, weight, ToOneV, D, W2_1, W2_2, W2_3, Eigens2_1, Eigens2_2, ComDevice, DiffLossSwitch):
    batchsize = u.size(0)
    nt = u.size(1)

    Du_flex1, Du_flex2, Du_rigid1, Du_rigid2, Du_Magnitude, Derivative_Data = FDD_HM(u, x, W2_1, W2_2, W2_3, Eigens2_1, Eigens2_2, weight, ToOneV, D, ComDevice, DiffLossSwitch)
    # criterion = torch.nn.SmoothL1Loss()
    criterion = torch.nn.L1Loss()
    loss_f = criterion(Du_flex1, torch.zeros_like(Du_flex1)) + criterion(Du_flex2, torch.zeros_like(Du_flex2)) + \
             criterion(Du_rigid1, torch.zeros_like(Du_rigid1)) + criterion(Du_rigid2, torch.zeros_like(Du_rigid2))
    return loss_f, Du_Magnitude, Derivative_Data


def Var_Encoder_HM(x, u, DOF_flex):
    U_flex = u[:, :, :DOF_flex]  # Output of flexible modes
    U_rigid = u[:, :, DOF_flex:]  # Output of rigid motions
    Excitation = x[:, 1:-1, 1:]  # Excitations
    return Excitation, U_flex, U_rigid


def ToOne_Action_HM(ToOneV, x, u, D):
    # Excitation, U_flex, U_rigid = Var_Encoder_HM(x, u, DOF_flex)
    # ReNormalization for input and output
    MeanV_in, StdV_in = ToOneV[0, :(1 + 3 + 6), 0], ToOneV[0, :(1 + 3 + 6), 3]
    MeanV_out, StdV_out = ToOneV[0, 10:(10 + 15 + 10 + 18), 0], ToOneV[0, 10:(10 + 15 + 10 + 18), 3]
    x = x * StdV_in + MeanV_in
    u = u * StdV_out + MeanV_out
    # Extract information
    Excitation = x[:, 1:-1, 1:(1 + 3)]
    K1, K2, C1, C2, M1, M2 = x[:, 0, 4], x[:, 0, 5], x[:, 0, 6], x[:, 0, 7], x[:, 0, 8], x[:, 0, 9]
    K1 = K1.unsqueeze(-1).unsqueeze(-1)
    K2 = K2.unsqueeze(-1).unsqueeze(-1)
    C1 = C1.unsqueeze(-1).unsqueeze(-1)
    C2 = C2.unsqueeze(-1).unsqueeze(-1)
    M1 = M1.unsqueeze(-1).unsqueeze(-1)
    M2 = M2.unsqueeze(-1).unsqueeze(-1)
    U_flex1, U_flex2, U_rigid1, U_rigid2 = u[:, :, :15], u[:, :, 15:25], u[:, :, 25:34], u[:, :, 34:]
    # Derivatives computation
    nt = u.size(1)
    dt = D / nt
    dU_rigid1 = (U_rigid1[:, 2:, :] - U_rigid1[:, :-2, :]) / (2 * dt)
    ddU_rigid1 = (U_rigid1[:, 2:, :] - 2 * U_rigid1[:, 1:-1, :] + U_rigid1[:, :-2, :]) / (dt ** 2)
    dU_rigid2 = (U_rigid2[:, 2:, :] - U_rigid2[:, :-2, :]) / (2 * dt)
    ddU_rigid2 = (U_rigid2[:, 2:, :] - 2 * U_rigid2[:, 1:-1, :] + U_rigid2[:, :-2, :]) / (dt ** 2)
    dU_flex1 = (U_flex1[:, 2:, :] - U_flex1[:, :-2, :]) / (2 * dt)
    ddU_flex1 = (U_flex1[:, 2:, :] - 2 * U_flex1[:, 1:-1, :] + U_flex1[:, :-2, :]) / (dt ** 2)
    dU_flex2 = (U_flex2[:, 2:, :] - U_flex2[:, :-2, :]) / (2 * dt)
    ddU_flex2 = (U_flex2[:, 2:, :] - 2 * U_flex2[:, 1:-1, :] + U_flex2[:, :-2, :]) / (dt ** 2)
    # Reshape the solution of the system
    U_flex1, U_flex2, U_rigid1, U_rigid2 = U_flex1[:, 1:-1, :], U_flex2[:, 1:-1, :], U_rigid1[:, 1:-1, :], U_rigid2[:, 1:-1, :]

    return Excitation, K1, K2, C1, C2, M1, M2, U_flex1, U_flex2, U_rigid1, U_rigid2, dU_flex1, dU_flex2, dU_rigid1, dU_rigid2, ddU_flex1, ddU_flex2, ddU_rigid1, ddU_rigid2
