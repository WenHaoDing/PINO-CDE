import numpy as np
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import inspect
"""
PDE loss Subroutine for Vehicle-Track Coupled Dynamics (VTCD). This subroutine
(1). generates the Derivatives with a differential operation on the solutions;
(2). return the Derivatives of the signal outside for data loss computation;
(3). uses the Derivatives of the signal inside for PDE loss computation.
"""

def FDD_VTCD(u, x, weight, ToOneV, inputDim, outputDim, D, ComDevice, g=9.8):
    nt = u.size(1)
    nf = u.size(2)

    # Extraction and renormalization for all parameters
    E1, E2, E3, E4, Mc, Jc, Mt, Jt, Mw, Kp, Cp, Ks, Cs, Lc, Lt, Vc, Kkj, Ckj, \
    uZc, uBc, uZt1, uBt1, uZt2, uBt2, uZw1, uZw2, uZw3, uZw4, \
    duZc, duBc, duZt1, duBt1, duZt2, duBt2, duZw1, duZw2, duZw3, duZw4, \
    dduZc, dduBc, dduZt1, dduBt1, dduZt2, dduBt2, dduZw1, dduZw2, dduZw3, dduZw4, WIrre1, WIrre2, WIrre3, WIrre4 = ToOne_Action(ToOneV, x, u, D)

    # Encoding physics laws for the vehicle system dynamics
    Du1 = Mc * g - 2 * Cs * duZc - 2 * Ks * uZc + Cs * duZt1 + Ks * uZt1 + Cs * duZt2 + Ks * uZt2
    Du2 = -2 * Cs * Lc ** 2 * duBc - 2 * Ks * Lc ** 2 * uBc - Cs * Lc * duZt1 + Cs * Lc * duZt2 - Ks * Lc * uZt1 + Ks * Lc * uZt2
    Du3 = Mt * g - (2 * Cp + Cs) * duZt1 - (
            2 * Kp + Ks) * uZt1 + Cs * duZc + Ks * uZc + Cp * duZw1 + Cp * duZw2 + Kp * uZw1 + Kp * uZw2 - Cs * Lc * duBc - Ks * Lc * uBc
    Du4 = -2 * Cp * Lt ** 2 * duBt1 - 2 * Kp * Lt ** 2 * uBt1 - Cp * Lt * duZw1 + Cp * Lt * duZw2 - Kp * Lt * uZw1 + Kp * Lt * uZw2
    Du5 = Mt * g - (2 * Cp + Cs) * duZt2 - (
            2 * Kp + Ks) * uZt2 + Cs * duZc + Ks * uZc + Cp * duZw3 + Cp * duZw4 + Kp * uZw3 + Kp * uZw4 + Cs * Lc * duBc + Ks * Lc * uBc
    Du6 = -2 * Cp * Lt ** 2 * duBt2 - 2 * Kp * Lt ** 2 * uBt2 - Cp * Lt * duZw3 + Cp * Lt * duZw4 - Kp * Lt * uZw3 + Kp * Lt * uZw4
    Du7 = -Kp * uZw1 + Cp * duZt1 + Kp * uZt1 - Cp * Lt * duBt1 - Kp * Lt * uBt1 + Mw * g
    Du8 = -Kp * uZw2 + Cp * duZt1 + Kp * uZt1 + Cp * Lt * duBt1 + Kp * Lt * uBt1 + Mw * g
    Du9 = -Kp * uZw3 + Cp * duZt2 + Kp * uZt2 - Cp * Lt * duBt2 - Kp * Lt * uBt2 + Mw * g
    Du10 = -Kp * uZw4 + Cp * duZt2 + Kp * uZt2 + Cp * Lt * duBt2 + Kp * Lt * uBt2 + Mw * g
    # Apply Static Force
    SF_Pre = (2 * Mt + Mc) * g / 4
    SF_Sec = Mc * g / 2
    Du1 += - 2 * SF_Sec
    Du3 += 1 * SF_Sec - 2 * SF_Pre
    Du5 += 1 * SF_Sec - 2 * SF_Pre
    Du7 += 1 * SF_Pre
    Du8 += 1 * SF_Pre
    Du9 += 1 * SF_Pre
    Du10 += 1 * SF_Pre
    # Apply Inertia Force
    Du1 -= Mc * dduZc
    Du2 -= Jc * dduBc
    Du3 -= Mt * dduZt1
    Du4 -= Jt * dduBt1
    Du5 -= Mt * dduZt2
    Du6 -= Jt * dduBt2
    Du7 -= Mw * dduZw1
    Du8 -= Mw * dduZw2
    Du9 -= Mw * dduZw3
    Du10 -= Mw * dduZw4
    # Apply Wheel Force Excitation
    G = 4.57 * 0.43 ** (-0.149) * 1e-8
    detZ1 = uZw1 - WIrre1 - E1
    detZ2 = uZw2 - WIrre2 - E2
    detZ3 = uZw3 - WIrre3 - E3
    detZ4 = uZw4 - WIrre4 - E4
    # Once the wheel gets off from the rail, there will be zero wheel-rail force.
    Du7 -= ((1 / G) * detZ1.clamp(min=0.0)) ** (3 / 2)
    Du8 -= ((1 / G) * detZ2.clamp(min=0.0)) ** (3 / 2)
    Du9 -= ((1 / G) * detZ3.clamp(min=0.0)) ** (3 / 2)
    Du10 -= ((1 / G) * detZ4.clamp(min=0.0)) ** (3 / 2)

    Du_Magnitude = torch.zeros([1, 20])
    weight = weight.unsqueeze(-1)
    TargetMin = 0.025
    Du1 = 1 * TargetMin / weight[:, 0, :].to(ComDevice) * Du1
    Du2 = 1 * TargetMin / weight[:, 1, :].to(ComDevice) * Du2
    Du3 = 1 * TargetMin / weight[:, 2, :].to(ComDevice) * Du3
    Du4 = 1 * TargetMin / weight[:, 3, :].to(ComDevice) * Du4
    Du5 = 1 * TargetMin / weight[:, 4, :].to(ComDevice) * Du5
    Du6 = 1 * TargetMin / weight[:, 5, :].to(ComDevice) * Du6
    Du7 = 1 * TargetMin / weight[:, 6, :].to(ComDevice) * Du7
    Du8 = 1 * TargetMin / weight[:, 7, :].to(ComDevice) * Du8
    Du9 = 1 * TargetMin / weight[:, 8, :].to(ComDevice) * Du9
    Du10 = 1 * TargetMin / weight[:, 9, :].to(ComDevice) * Du10

    # Storing the std information for Du computation
    Du_Magnitude[0, 0] = torch.std(Du1)
    Du_Magnitude[0, 1] = torch.std(Du2)
    Du_Magnitude[0, 2] = torch.std(Du3)
    Du_Magnitude[0, 3] = torch.std(Du4)
    Du_Magnitude[0, 4] = torch.std(Du5)
    Du_Magnitude[0, 5] = torch.std(Du6)
    Du_Magnitude[0, 6] = torch.std(Du7)
    Du_Magnitude[0, 7] = torch.std(Du8)
    Du_Magnitude[0, 8] = torch.std(Du9)
    Du_Magnitude[0, 9] = torch.std(Du10)
    # Storing the mean information for Du computation
    Du_Magnitude[0, 10] = torch.mean(torch.abs(Du1))
    Du_Magnitude[0, 11] = torch.mean(torch.abs(Du2))
    Du_Magnitude[0, 12] = torch.mean(torch.abs(Du3))
    Du_Magnitude[0, 13] = torch.mean(torch.abs(Du4))
    Du_Magnitude[0, 14] = torch.mean(torch.abs(Du5))
    Du_Magnitude[0, 15] = torch.mean(torch.abs(Du6))
    Du_Magnitude[0, 16] = torch.mean(torch.abs(Du7))
    Du_Magnitude[0, 17] = torch.mean(torch.abs(Du8))
    Du_Magnitude[0, 18] = torch.mean(torch.abs(Du9))
    Du_Magnitude[0, 19] = torch.mean(torch.abs(Du10))

    # Regenerate the data tensor for derivatives
    MeanV, StdV = ToOneV[0, :, 0], ToOneV[0, :, 3]
    duZc = (duZc - MeanV[29]) / StdV[29]
    duBc = (duBc - MeanV[30]) / StdV[30]
    duZt1 = (duZt1 - MeanV[31]) / StdV[31]
    duBt1 = (duBt1 - MeanV[32]) / StdV[32]
    duZt2 = (duZt2 - MeanV[33]) / StdV[33]
    duBt2 = (duBt2 - MeanV[34]) / StdV[34]
    duZw1 = (duZw1 - MeanV[35]) / StdV[35]
    duZw2 = (duZw2 - MeanV[36]) / StdV[36]
    duZw3 = (duZw3 - MeanV[37]) / StdV[37]
    duZw4 = (duZw4 - MeanV[38]) / StdV[38]
    dduZc = (dduZc - MeanV[39]) / StdV[39]
    dduBc = (dduBc - MeanV[40]) / StdV[40]
    dduZt1 = (dduZt1 - MeanV[41]) / StdV[41]
    dduBt1 = (dduBt1 - MeanV[42]) / StdV[42]
    dduZt2 = (dduZt2 - MeanV[43]) / StdV[43]
    dduBt2 = (dduBt2 - MeanV[44]) / StdV[44]
    dduZw1 = (dduZw1 - MeanV[45]) / StdV[45]
    dduZw2 = (dduZw2 - MeanV[46]) / StdV[46]
    dduZw3 = (dduZw3 - MeanV[47]) / StdV[47]
    dduZw4 = (dduZw4 - MeanV[48]) / StdV[48]
    Derivative_Data = torch.stack([duZc, duBc, duZt1, duBt1, duZt2, duBt2, duZw1, duZw2, duZw3, duZw4, dduZc, dduBc, dduZt1, dduBt1, dduZt2, dduBt2, dduZw1, dduZw2, dduZw3, dduZw4], dim=-1)
    return Du1, Du2, Du3, Du4, Du5, Du6, Du7, Du8, Du9, Du10, Du_Magnitude, Derivative_Data


def VTCD_PINO_loss(u, x, weight, ToOneV, inputDim, outputDim, D, ComDevice):
    batchsize = u.size(0)
    nt = u.size(1)
    u = u.reshape(batchsize, nt, outputDim)
    # lploss = LpLoss(size_average=True)

    weights = torch.tensor([0, 1])
    # Du, Du_Magnitude = FDD_VTCD(u, x, ToOneV, inputDim, outputDim)
    Du1, Du2, Du3, Du4, Du5, Du6, Du7, Du8, Du9, Du10, Du_Magnitude, Derivative_Data = FDD_VTCD(u, x, weight, ToOneV, inputDim, outputDim, D, ComDevice)
    f = torch.zeros(Du1.shape, device=u.device)
    # loss_f = F.mse_loss(Du1, f) + F.mse_loss(Du2, f) + F.mse_loss(Du3, f) + F.mse_loss(Du4, f) + F.mse_loss(Du5, f) + F.mse_loss(Du6, f) + F.mse_loss(Du7, f) + F.mse_loss(Du8, f) + F.mse_loss(Du9, f) + F.mse_loss(Du10, f)
    # criterion = torch.nn.SmoothL1Loss()
    criterion = torch.nn.L1Loss()
    loss_f = criterion(Du1, f) + criterion(Du2, f) + criterion(Du3, f) + criterion(Du4, f) + criterion(Du5, f) + criterion(Du6, f) + criterion(Du7, f) + criterion(Du8, f) + criterion(Du9, f) + criterion(Du10, f)
    return loss_f, Du_Magnitude, Derivative_Data


def Var_Encoder(x, u):
    P = x[:, 1:-1, 5:5 + 14]  # Parameters of the DOF system
    E = x[:, 1:-1, 1:5]  # Parameters of the Excitation
    U = u[:, :, :]  # Network output
    # System Parameters
    Mc, Jc, Mt, Jt, Mw, Kp, Cp, Ks, Cs, Lc, Lt, Vc, Kkj, Ckj = P[:, :, 0], P[:, :, 1], P[:, :, 2], P[:, :, 3], P[:, :, 4], P[:, :, 5], P[:, :, 6], P[:, :, 7], P[:, :, 8], P[:, :, 9], P[:, :, 10], P[:, :, 11], P[:, :, 12], P[:, :, 13]
    E1, E2, E3, E4 = E[:, :, 0], E[:, :, 1], E[:, :, 2], E[:, :, 3]
    uZc, uBc, uZt1, uBt1, uZt2, uBt2, uZw1, uZw2, uZw3, uZw4 = U[:, :, 0], U[:, :, 1], U[:, :, 2], U[:, :, 3], U[:, :, 4], U[:, :, 5], U[:, :, 6], U[:, :, 7], U[:, :, 8], U[:, :, 9]
    WIrre1, WIrre2, WIrre3, WIrre4 = U[:, :, -4], U[:, :, -3], U[:, :, -2], U[:, :, -1]
    return E1, E2, E3, E4, Mc, Jc, Mt, Jt, Mw, Kp, Cp, Ks, Cs, Lc, Lt, Vc, Kkj, Ckj, uZc, uBc, uZt1, uBt1, uZt2, uBt2, uZw1, uZw2, uZw3, uZw4, WIrre1, WIrre2, WIrre3, WIrre4


def ToOne_Action(ToOneV, x, u, D):
    E1, E2, E3, E4, Mc, Jc, Mt, Jt, Mw, Kp, Cp, Ks, Cs, Lc, Lt, Vc, Kkj, Ckj, uZc, uBc, uZt1, uBt1, uZt2, uBt2, uZw1, uZw2, uZw3, uZw4, WIrre1, WIrre2, WIrre3, WIrre4 = Var_Encoder(x, u)

    MeanV, StdV = ToOneV[0, :, 0], ToOneV[0, :, 3]
    E1 = E1 * StdV[1] + MeanV[1]
    E2 = E2 * StdV[2] + MeanV[2]
    E3 = E3 * StdV[3] + MeanV[3]
    E4 = E4 * StdV[4] + MeanV[4]
    Mc = Mc * StdV[5] + MeanV[5]
    Jc = Jc * StdV[6] + MeanV[6]
    Mt = Mt * StdV[7] + MeanV[7]
    Jt = Jt * StdV[8] + MeanV[8]
    Mw = Mw * StdV[9] + MeanV[9]
    Kp = Kp * StdV[10] + MeanV[10]
    Cp = Cp * StdV[11] + MeanV[11]
    Ks = Ks * StdV[12] + MeanV[12]
    Cs = Cs * StdV[13] + MeanV[13]
    Lc = Lc * StdV[14] + MeanV[14]
    Lt = Lt * StdV[15] + MeanV[15]
    Vc = Vc * StdV[16] + MeanV[16]
    Kkj = Kkj * StdV[17] + MeanV[17]
    Ckj = Ckj * StdV[18] + MeanV[18]
    uZc = uZc * StdV[19] + MeanV[19]
    uBc = uBc * StdV[20] + MeanV[20]
    uZt1 = uZt1 * StdV[21] + MeanV[21]
    uBt1 = uBt1 * StdV[22] + MeanV[22]
    uZt2 = uZt2 * StdV[23] + MeanV[23]
    uBt2 = uBt2 * StdV[24] + MeanV[24]
    uZw1 = uZw1 * StdV[25] + MeanV[25]
    uZw2 = uZw2 * StdV[26] + MeanV[26]
    uZw3 = uZw3 * StdV[27] + MeanV[27]
    uZw4 = uZw4 * StdV[28] + MeanV[28]
    WIrre1 = WIrre1 * StdV[-4] + MeanV[-4]
    WIrre2 = WIrre2 * StdV[-3] + MeanV[-3]
    WIrre3 = WIrre3 * StdV[-2] + MeanV[-2]
    WIrre4 = WIrre4 * StdV[-1] + MeanV[-1]

    # Perform derivative computation
    nt = u.size(1)
    dt = D / nt
    # First Derivative of the system
    duZc = (uZc[:, 2:] - uZc[:, :-2]) / (2 * dt)
    duBc = (uBc[:, 2:] - uBc[:, :-2]) / (2 * dt)
    duZt1 = (uZt1[:, 2:] - uZt1[:, :-2]) / (2 * dt)
    duBt1 = (uBt1[:, 2:] - uBt1[:, :-2]) / (2 * dt)
    duZt2 = (uZt2[:, 2:] - uZt2[:, :-2]) / (2 * dt)
    duBt2 = (uBt2[:, 2:] - uBt2[:, :-2]) / (2 * dt)
    duZw1 = (uZw1[:, 2:] - uZw1[:, :-2]) / (2 * dt)
    duZw2 = (uZw2[:, 2:] - uZw2[:, :-2]) / (2 * dt)
    duZw3 = (uZw3[:, 2:] - uZw3[:, :-2]) / (2 * dt)
    duZw4 = (uZw4[:, 2:] - uZw4[:, :-2]) / (2 * dt)
    # Second Derivative of the system
    dduZc = (uZc[:, 2:] - 2 * uZc[:, 1:-1] + uZc[:, :-2]) / (dt ** 2)
    dduBc = (uBc[:, 2:] - 2 * uBc[:, 1:-1] + uBc[:, :-2]) / (dt ** 2)
    dduZt1 = (uZt1[:, 2:] - 2 * uZt1[:, 1:-1] + uZt1[:, :-2]) / (dt ** 2)
    dduBt1 = (uBt1[:, 2:] - 2 * uBt1[:, 1:-1] + uBt1[:, :-2]) / (dt ** 2)
    dduZt2 = (uZt2[:, 2:] - 2 * uZt2[:, 1:-1] + uZt2[:, :-2]) / (dt ** 2)
    dduBt2 = (uBt2[:, 2:] - 2 * uBt2[:, 1:-1] + uBt2[:, :-2]) / (dt ** 2)
    dduZw1 = (uZw1[:, 2:] - 2 * uZw1[:, 1:-1] + uZw1[:, :-2]) / (dt ** 2)
    dduZw2 = (uZw2[:, 2:] - 2 * uZw2[:, 1:-1] + uZw2[:, :-2]) / (dt ** 2)
    dduZw3 = (uZw3[:, 2:] - 2 * uZw3[:, 1:-1] + uZw3[:, :-2]) / (dt ** 2)
    dduZw4 = (uZw4[:, 2:] - 2 * uZw4[:, 1:-1] + uZw4[:, :-2]) / (dt ** 2)
    # Reshape the solution of the system
    uZc = uZc[:, 1:-1]
    uBc = uBc[:, 1:-1]
    uZt1 = uZt1[:, 1:-1]
    uBt1 = uBt1[:, 1:-1]
    uZt2 = uZt2[:, 1:-1]
    uBt2 = uBt2[:, 1:-1]
    uZw1 = uZw1[:, 1:-1]
    uZw2 = uZw2[:, 1:-1]
    uZw3 = uZw3[:, 1:-1]
    uZw4 = uZw4[:, 1:-1]
    WIrre1 = WIrre1[:, 1:-1]
    WIrre2 = WIrre2[:, 1:-1]
    WIrre3 = WIrre3[:, 1:-1]
    WIrre4 = WIrre4[:, 1:-1]

    return E1, E2, E3, E4, Mc, Jc, Mt, Jt, Mw, Kp, Cp, Ks, Cs, Lc, Lt, Vc, Kkj, Ckj, \
           uZc, uBc, uZt1, uBt1, uZt2, uBt2, uZw1, uZw2, uZw3, uZw4, \
           duZc, duBc, duZt1, duBt1, duZt2, duBt2, duZw1, duZw2, duZw3, duZw4, \
           dduZc, dduBc, dduZt1, dduBt1, dduZt2, dduBt2, dduZw1, dduZw2, dduZw3, dduZw4, \
           WIrre1, WIrre2, WIrre3, WIrre4


"""
PDE loss Subroutine for VTCD (Without ODE Weights)
(1). Input model output and input, generating Derivatives of the original signal
(2). Return the Derivatives of the signal outside for data loss computation.
(3). Using the Derivatives of the signal inside for PDE loss computation.
"""


def FDD_VTCD_Variant1(u, x, ToOneV, inputDim, outputDim, D, g=9.8):
    nt = u.size(1)
    nf = u.size(2)

    # Extracting and ToOne actions for all parameters
    E1, E2, E3, E4, Mc, Jc, Mt, Jt, Mw, Kp, Cp, Ks, Cs, Lc, Lt, Vc, Kkj, Ckj, \
    uZc, uBc, uZt1, uBt1, uZt2, uBt2, uZw1, uZw2, uZw3, uZw4, \
    duZc, duBc, duZt1, duBt1, duZt2, duBt2, duZw1, duZw2, duZw3, duZw4, \
    dduZc, dduBc, dduZt1, dduBt1, dduZt2, dduBt2, dduZw1, dduZw2, dduZw3, dduZw4, WIrre1, WIrre2, WIrre3, WIrre4 = ToOne_Action_Variant1(ToOneV, x, u, D)

    # Encoding physics laws for the vehicle system
    Du1 = Mc * g - 2 * Cs * duZc - 2 * Ks * uZc + Cs * duZt1 + Ks * uZt1 + Cs * duZt2 + Ks * uZt2
    Du2 = -2 * Cs * Lc ** 2 * duBc - 2 * Ks * Lc ** 2 * uBc - Cs * Lc * duZt1 + Cs * Lc * duZt2 - Ks * Lc * uZt1 + Ks * Lc * uZt2
    Du3 = Mt * g - (2 * Cp + Cs) * duZt1 - (
            2 * Kp + Ks) * uZt1 + Cs * duZc + Ks * uZc + Cp * duZw1 + Cp * duZw2 + Kp * uZw1 + Kp * uZw2 - Cs * Lc * duBc - Ks * Lc * uBc
    Du4 = -2 * Cp * Lt ** 2 * duBt1 - 2 * Kp * Lt ** 2 * uBt1 - Cp * Lt * duZw1 + Cp * Lt * duZw2 - Kp * Lt * uZw1 + Kp * Lt * uZw2
    Du5 = Mt * g - (2 * Cp + Cs) * duZt2 - (
            2 * Kp + Ks) * uZt2 + Cs * duZc + Ks * uZc + Cp * duZw3 + Cp * duZw4 + Kp * uZw3 + Kp * uZw4 + Cs * Lc * duBc + Ks * Lc * uBc
    Du6 = -2 * Cp * Lt ** 2 * duBt2 - 2 * Kp * Lt ** 2 * uBt2 - Cp * Lt * duZw3 + Cp * Lt * duZw4 - Kp * Lt * uZw3 + Kp * Lt * uZw4
    Du7 = -Kp * uZw1 + Cp * duZt1 + Kp * uZt1 - Cp * Lt * duBt1 - Kp * Lt * uBt1 + Mw * g
    Du8 = -Kp * uZw2 + Cp * duZt1 + Kp * uZt1 + Cp * Lt * duBt1 + Kp * Lt * uBt1 + Mw * g
    Du9 = -Kp * uZw3 + Cp * duZt2 + Kp * uZt2 - Cp * Lt * duBt2 - Kp * Lt * uBt2 + Mw * g
    Du10 = -Kp * uZw4 + Cp * duZt2 + Kp * uZt2 + Cp * Lt * duBt2 + Kp * Lt * uBt2 + Mw * g
    # Apply Static Force
    SF_Pre = (2 * Mt + Mc) * g / 4
    SF_Sec = Mc * g / 2
    Du1 += - 2 * SF_Sec
    Du3 += 1 * SF_Sec - 2 * SF_Pre
    Du5 += 1 * SF_Sec - 2 * SF_Pre
    Du7 += 1 * SF_Pre
    Du8 += 1 * SF_Pre
    Du9 += 1 * SF_Pre
    Du10 += 1 * SF_Pre
    # Apply Inertia Force
    Du1 -= Mc * dduZc
    Du2 -= Jc * dduBc
    Du3 -= Mt * dduZt1
    Du4 -= Jt * dduBt1
    Du5 -= Mt * dduZt2
    Du6 -= Jt * dduBt2
    Du7 -= Mw * dduZw1
    Du8 -= Mw * dduZw2
    Du9 -= Mw * dduZw3
    Du10 -= Mw * dduZw4
    # Apply Wheel Force Excitation
    G = 4.57 * 0.43 ** (-0.149) * 1e-8
    detZ1 = uZw1 - WIrre1 - E1
    detZ2 = uZw2 - WIrre2 - E2
    detZ3 = uZw3 - WIrre3 - E3
    detZ4 = uZw4 - WIrre4 - E4
    # Non-linear hertz contact equations (see page 55 in ref[33])
    Du7 -= ((1 / G) * detZ1.clamp(min=0.0)) ** (3 / 2)
    Du8 -= ((1 / G) * detZ2.clamp(min=0.0)) ** (3 / 2)
    Du9 -= ((1 / G) * detZ3.clamp(min=0.0)) ** (3 / 2)
    Du10 -= ((1 / G) * detZ4.clamp(min=0.0)) ** (3 / 2)
    # Compounding with weights
    # weights = 0.025 / torch.tensor([100, 1000, 100, 350, 100, 350, 4000, 4000, 4000, 4000])
    weights = 0.025 / torch.tensor([55, 350, 150, 100, 140, 100, 14000, 14000, 14000, 14000])
    Du_Magnitude = torch.zeros([10])
    Du_List = [Du1, Du2, Du3, Du4, Du5, Du6, Du7, Du8, Du9, Du10]
    Du_Index = 0

    for DuTarget in Du_List:
        DuTarget *= weights[Du_Index]
        # Du_Magnitude[Du_Index] = torch.max(torch.abs(DuTarget))
        Du_Index += 1

    # Regenerate the data tensor for derivatives
    MeanV, StdV = ToOneV[0, :, 0], ToOneV[0, :, 3]
    duZc = (duZc - MeanV[29]) / StdV[29]
    duBc = (duBc - MeanV[30]) / StdV[30]
    duZt1 = (duZt1 - MeanV[31]) / StdV[31]
    duBt1 = (duBt1 - MeanV[32]) / StdV[32]
    duZt2 = (duZt2 - MeanV[33]) / StdV[33]
    duBt2 = (duBt2 - MeanV[34]) / StdV[34]
    duZw1 = (duZw1 - MeanV[35]) / StdV[35]
    duZw2 = (duZw2 - MeanV[36]) / StdV[36]
    duZw3 = (duZw3 - MeanV[37]) / StdV[37]
    duZw4 = (duZw4 - MeanV[38]) / StdV[38]
    dduZc = (dduZc - MeanV[39]) / StdV[39]
    dduBc = (dduBc - MeanV[40]) / StdV[40]
    dduZt1 = (dduZt1 - MeanV[41]) / StdV[41]
    dduBt1 = (dduBt1 - MeanV[42]) / StdV[42]
    dduZt2 = (dduZt2 - MeanV[43]) / StdV[43]
    dduBt2 = (dduBt2 - MeanV[44]) / StdV[44]
    dduZw1 = (dduZw1 - MeanV[45]) / StdV[45]
    dduZw2 = (dduZw2 - MeanV[46]) / StdV[46]
    dduZw3 = (dduZw3 - MeanV[47]) / StdV[47]
    dduZw4 = (dduZw4 - MeanV[48]) / StdV[48]
    Derivative_Data = torch.stack([duZc, duBc, duZt1, duBt1, duZt2, duBt2, duZw1, duZw2, duZw3, duZw4, dduZc, dduBc, dduZt1, dduBt1, dduZt2, dduBt2, dduZw1, dduZw2, dduZw3, dduZw4], dim=-1)
    return Du1, Du2, Du3, Du4, Du5, Du6, Du7, Du8, Du9, Du10, Du_Magnitude, Derivative_Data


def VTCD_PINO_loss_Variant1(u, x, ToOneV, inputDim, outputDim, D):
    batchsize = u.size(0)
    nt = u.size(1)
    u = u.reshape(batchsize, nt, outputDim)
    # lploss = LpLoss(size_average=True)

    weights = torch.tensor([0, 1])
    # Du, Du_Magnitude = FDD_VTCD(u, x, ToOneV, inputDim, outputDim)
    Du1, Du2, Du3, Du4, Du5, Du6, Du7, Du8, Du9, Du10, Du_Magnitude, Derivative_Data = FDD_VTCD_Variant1(u, x, ToOneV, inputDim, outputDim, D)
    f = torch.zeros(Du1.shape, device=u.device)
    # loss_f = F.mse_loss(Du1, f) + F.mse_loss(Du2, f) + F.mse_loss(Du3, f) + F.mse_loss(Du4, f) + F.mse_loss(Du5, f) + F.mse_loss(Du6, f) + F.mse_loss(Du7, f) + F.mse_loss(Du8, f) + F.mse_loss(Du9, f) + F.mse_loss(Du10, f)
    criterion = torch.nn.SmoothL1Loss()
    loss_f = criterion(Du1, f) + criterion(Du2, f) + criterion(Du3, f) + criterion(Du4, f) + criterion(Du5, f) + criterion(Du6, f) + criterion(Du7, f) + criterion(Du8, f) + criterion(Du9, f) + criterion(Du10, f)
    return loss_f, Du_Magnitude, Derivative_Data


def Var_Encoder_Variant1(x, u):
    P = x[:, 1:-1, 5:5 + 14]  # Parameters of the DOF system
    E = x[:, 1:-1, 1:5]  # Parameters of the Excitation
    U = u[:, :, :]  # Network output
    # System Parameters
    Mc, Jc, Mt, Jt, Mw, Kp, Cp, Ks, Cs, Lc, Lt, Vc, Kkj, Ckj = P[:, :, 0], P[:, :, 1], P[:, :, 2], P[:, :, 3], P[:, :, 4], P[:, :, 5], P[:, :, 6], P[:, :, 7], P[:, :, 8], P[:, :, 9], P[:, :, 10], P[:, :, 11], P[:, :, 12], P[:, :, 13]
    E1, E2, E3, E4 = E[:, :, 0], E[:, :, 1], E[:, :, 2], E[:, :, 3]
    uZc, uBc, uZt1, uBt1, uZt2, uBt2, uZw1, uZw2, uZw3, uZw4 = U[:, :, 0], U[:, :, 1], U[:, :, 2], U[:, :, 3], U[:, :, 4], U[:, :, 5], U[:, :, 6], U[:, :, 7], U[:, :, 8], U[:, :, 9]
    WIrre1, WIrre2, WIrre3, WIrre4 = U[:, :, -4], U[:, :, -3], U[:, :, -2], U[:, :, -1]
    return E1, E2, E3, E4, Mc, Jc, Mt, Jt, Mw, Kp, Cp, Ks, Cs, Lc, Lt, Vc, Kkj, Ckj, uZc, uBc, uZt1, uBt1, uZt2, uBt2, uZw1, uZw2, uZw3, uZw4, WIrre1, WIrre2, WIrre3, WIrre4


def ToOne_Action_Variant1(ToOneV, x, u, D):
    E1, E2, E3, E4, Mc, Jc, Mt, Jt, Mw, Kp, Cp, Ks, Cs, Lc, Lt, Vc, Kkj, Ckj, uZc, uBc, uZt1, uBt1, uZt2, uBt2, uZw1, uZw2, uZw3, uZw4, WIrre1, WIrre2, WIrre3, WIrre4 = Var_Encoder_Variant1(x, u)

    MeanV, StdV = ToOneV[0, :, 0], ToOneV[0, :, 3]
    E1 = E1 * StdV[1] + MeanV[1]
    E2 = E2 * StdV[2] + MeanV[2]
    E3 = E3 * StdV[3] + MeanV[3]
    E4 = E4 * StdV[4] + MeanV[4]
    Mc = Mc * StdV[5] + MeanV[5]
    Jc = Jc * StdV[6] + MeanV[6]
    Mt = Mt * StdV[7] + MeanV[7]
    Jt = Jt * StdV[8] + MeanV[8]
    Mw = Mw * StdV[9] + MeanV[9]
    Kp = Kp * StdV[10] + MeanV[10]
    Cp = Cp * StdV[11] + MeanV[11]
    Ks = Ks * StdV[12] + MeanV[12]
    Cs = Cs * StdV[13] + MeanV[13]
    Lc = Lc * StdV[14] + MeanV[14]
    Lt = Lt * StdV[15] + MeanV[15]
    Vc = Vc * StdV[16] + MeanV[16]
    Kkj = Kkj * StdV[17] + MeanV[17]
    Ckj = Ckj * StdV[18] + MeanV[18]
    uZc = uZc * StdV[19] + MeanV[19]
    uBc = uBc * StdV[20] + MeanV[20]
    uZt1 = uZt1 * StdV[21] + MeanV[21]
    uBt1 = uBt1 * StdV[22] + MeanV[22]
    uZt2 = uZt2 * StdV[23] + MeanV[23]
    uBt2 = uBt2 * StdV[24] + MeanV[24]
    uZw1 = uZw1 * StdV[25] + MeanV[25]
    uZw2 = uZw2 * StdV[26] + MeanV[26]
    uZw3 = uZw3 * StdV[27] + MeanV[27]
    uZw4 = uZw4 * StdV[28] + MeanV[28]
    WIrre1 = WIrre1 * StdV[-4] + MeanV[-4]
    WIrre2 = WIrre2 * StdV[-3] + MeanV[-3]
    WIrre3 = WIrre3 * StdV[-2] + MeanV[-2]
    WIrre4 = WIrre4 * StdV[-1] + MeanV[-1]

    # Perform derivative computation
    nt = u.size(1)
    dt = D / nt
    # First Derivative of the system
    duZc = (uZc[:, 2:] - uZc[:, :-2]) / (2 * dt)
    duBc = (uBc[:, 2:] - uBc[:, :-2]) / (2 * dt)
    duZt1 = (uZt1[:, 2:] - uZt1[:, :-2]) / (2 * dt)
    duBt1 = (uBt1[:, 2:] - uBt1[:, :-2]) / (2 * dt)
    duZt2 = (uZt2[:, 2:] - uZt2[:, :-2]) / (2 * dt)
    duBt2 = (uBt2[:, 2:] - uBt2[:, :-2]) / (2 * dt)
    duZw1 = (uZw1[:, 2:] - uZw1[:, :-2]) / (2 * dt)
    duZw2 = (uZw2[:, 2:] - uZw2[:, :-2]) / (2 * dt)
    duZw3 = (uZw3[:, 2:] - uZw3[:, :-2]) / (2 * dt)
    duZw4 = (uZw4[:, 2:] - uZw4[:, :-2]) / (2 * dt)
    # Second Derivative of the system
    dduZc = (uZc[:, 2:] - 2 * uZc[:, 1:-1] + uZc[:, :-2]) / (dt ** 2)
    dduBc = (uBc[:, 2:] - 2 * uBc[:, 1:-1] + uBc[:, :-2]) / (dt ** 2)
    dduZt1 = (uZt1[:, 2:] - 2 * uZt1[:, 1:-1] + uZt1[:, :-2]) / (dt ** 2)
    dduBt1 = (uBt1[:, 2:] - 2 * uBt1[:, 1:-1] + uBt1[:, :-2]) / (dt ** 2)
    dduZt2 = (uZt2[:, 2:] - 2 * uZt2[:, 1:-1] + uZt2[:, :-2]) / (dt ** 2)
    dduBt2 = (uBt2[:, 2:] - 2 * uBt2[:, 1:-1] + uBt2[:, :-2]) / (dt ** 2)
    dduZw1 = (uZw1[:, 2:] - 2 * uZw1[:, 1:-1] + uZw1[:, :-2]) / (dt ** 2)
    dduZw2 = (uZw2[:, 2:] - 2 * uZw2[:, 1:-1] + uZw2[:, :-2]) / (dt ** 2)
    dduZw3 = (uZw3[:, 2:] - 2 * uZw3[:, 1:-1] + uZw3[:, :-2]) / (dt ** 2)
    dduZw4 = (uZw4[:, 2:] - 2 * uZw4[:, 1:-1] + uZw4[:, :-2]) / (dt ** 2)
    # Reshape the solution of the system
    uZc = uZc[:, 1:-1]
    uBc = uBc[:, 1:-1]
    uZt1 = uZt1[:, 1:-1]
    uBt1 = uBt1[:, 1:-1]
    uZt2 = uZt2[:, 1:-1]
    uBt2 = uBt2[:, 1:-1]
    uZw1 = uZw1[:, 1:-1]
    uZw2 = uZw2[:, 1:-1]
    uZw3 = uZw3[:, 1:-1]
    uZw4 = uZw4[:, 1:-1]
    WIrre1 = WIrre1[:, 1:-1]
    WIrre2 = WIrre2[:, 1:-1]
    WIrre3 = WIrre3[:, 1:-1]
    WIrre4 = WIrre4[:, 1:-1]

    return E1, E2, E3, E4, Mc, Jc, Mt, Jt, Mw, Kp, Cp, Ks, Cs, Lc, Lt, Vc, Kkj, Ckj, \
           uZc, uBc, uZt1, uBt1, uZt2, uBt2, uZw1, uZw2, uZw3, uZw4, \
           duZc, duBc, duZt1, duBt1, duZt2, duBt2, duZw1, duZw2, duZw3, duZw4, \
           dduZc, dduBc, dduZt1, dduBt1, dduZt2, dduBt2, dduZw1, dduZw2, dduZw3, dduZw4, \
           WIrre1, WIrre2, WIrre3, WIrre4

