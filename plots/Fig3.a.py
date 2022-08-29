import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import os

# The codes below will be used to generate parts of figures for Fig.1
plt.rc('font', family='Arial')
plt.rcParams['xtick.direction'] = 'in'
SavePath = 'E:/学习库/文章/PINO-MBD_Physics-Informed Neual Operator for multi-body dynamics/Nature文献/Draft/Figures/Fig3/'
Path = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/VTCDRunner'
FileName = Path + '/' + 'PINOMBD_Performance.txt'
Data = torch.tensor(np.loadtxt(FileName))
Data_Predict = Data[:, :14]
Data_GT = Data[:, 14:]
print(Data_GT.shape)
Path = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/data/Project_VTCD/Big_MCM_20000.mat'
StdV = torch.cat([torch.tensor(h5py.File(Path)['StdV']).to(torch.float32)[19:19+10, :].permute([1, 0]), torch.tensor(h5py.File(Path)['StdV']).to(torch.float32)[-4:, :].permute([1, 0])], dim=1)
MeanV = torch.cat([torch.tensor(h5py.File(Path)['MeanV']).to(torch.float32)[19:19+10, :].permute([1, 0]), torch.tensor(h5py.File(Path)['MeanV']).to(torch.float32)[-4:, :].permute([1, 0])], dim=1)

dt = 0.001
Data_GT = Data_GT * StdV + MeanV
Data_Predict = Data_Predict * StdV + MeanV
De1_GT = (Data_GT[2:, :] - Data_GT[:-2, :]) / (2 * dt)
De1_Predict = (Data_Predict[2:, :] - Data_Predict[:-2, :]) / (2 * dt)
De2_GT = (Data_GT[2:, :] - 2 * Data_GT[1:-1, :] + Data_GT[:-2, :]) / (dt ** 2)
De2_Predict = (Data_Predict[2:, :] - 2 * Data_Predict[1:-1, :] + Data_Predict[:-2, :]) / (dt ** 2)
axis1 = np.linspace(0, 4.5, 4500)
axis2 = np.linspace(0, 4.5, 4498)

Path = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/VTCDRunner'
FileName = Path + '/' + 'PINO_Performance.txt'
Data = torch.tensor(np.loadtxt(FileName))
Data_PINO = Data[:, :14]
Data_PINO = Data_PINO * StdV + MeanV
De1_PINO = (Data_PINO[2:, :] - Data_PINO[:-2, :]) / (2 * dt)
De2_PINO = (Data_PINO[2:, :] - 2 * Data_PINO[1:-1, :] + Data_PINO[:-2, :]) / (dt ** 2)

Path = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/VTCDRunner'
FileName = Path + '/' + 'FNO_Performance.txt'
Data = torch.tensor(np.loadtxt(FileName))
Data_FNO = Data[:, :14]
Data_FNO = Data_FNO * StdV + MeanV
De1_FNO = (Data_FNO[2:, :] - Data_FNO[:-2, :]) / (2 * dt)
De2_FNO = (Data_FNO[2:, :] - 2 * Data_FNO[1:-1, :] + Data_FNO[:-2, :]) / (dt ** 2)


for index in [0, 1, 2, 5, 6, 13]:
    if index == 0:
        ylabel1 = "Solution (mm)"
        ylabel2 = "$1^{st}$ derivative (m/s)"
        ylabel3 = "$2^{st}$ derivative (m/$s^2$)"
        f = 1e3
        Data_T1_GT, Data_T1_Pre = f * Data_GT[:, index], f * Data_Predict[:, index]
        Data_T2_GT, Data_T2_Pre = De1_GT[:, index], De1_Predict[:, index]
        Data_T3_GT, Data_T3_Pre = De2_GT[:, index], De2_Predict[:, index]

        Data_T1_PINO, Data_T1_FNO = f * Data_PINO[:, index], f * Data_FNO[:, index]
        Data_T2_PINO, Data_T2_FNO = De1_PINO[:, index], De1_FNO[:, index]
        Data_T3_PINO, Data_T3_FNO = De2_PINO[:, index], De2_FNO[:, index]
        label1 = "Ground truth"
        label2 = "With EN (V2)"
        label3 = "Without EN (V3)"
        label4 = "Physics-uninformed (V4)"
        PicName = 'Fig3a0.jpg'
        ylim = [-7.5, 4, -0.02, 0.02, -0.25, 0.25]
        Title = "Vertical vibration of carbody ($Z_c$)"

    if index == 1:
        ylabel1 = "Solution (rad)"
        ylabel2 = "$1^{st}$ derivative (rad/s)"
        ylabel3 = "$2^{st}$ derivative (rad/$s^2$)"
        f = 1
        Data_T1_GT, Data_T1_Pre = f * Data_GT[:, index], f * Data_Predict[:, index]
        Data_T2_GT, Data_T2_Pre = De1_GT[:, index], De1_Predict[:, index]
        Data_T3_GT, Data_T3_Pre = De2_GT[:, index], De2_Predict[:, index]

        Data_T1_PINO, Data_T1_FNO = f * Data_PINO[:, index], f * Data_FNO[:, index]
        Data_T2_PINO, Data_T2_FNO = De1_PINO[:, index], De1_FNO[:, index]
        Data_T3_PINO, Data_T3_FNO = De2_PINO[:, index], De2_FNO[:, index]
        label1 = "Ground truth"
        label2 = "With EN (V2)"
        label3 = "Prediction with PINO as backbone (Case3)"
        label4 = "Prediction with FNO as backbone  (Case1)"
        PicName = 'Fig3a1.jpg'
        ylim = [-0.0002, 0.0002, -0.002, 0.002, -0.04, 0.04]
        Title = "Rotation of carbody " + r"$\beta_c$"

    if index == 2:
        ylabel1 = "Solution (mm)"
        ylabel2 = "$1^{st}$ derivative (m/s)"
        ylabel3 = "$2^{st}$ derivative (m/$s^2$)"
        f = 1e3
        Data_T1_GT, Data_T1_Pre = f * Data_GT[:, index], f * Data_Predict[:, index]
        Data_T2_GT, Data_T2_Pre = De1_GT[:, index], De1_Predict[:, index]
        Data_T3_GT, Data_T3_Pre = De2_GT[:, index], De2_Predict[:, index]

        Data_T1_PINO, Data_T1_FNO = f * Data_PINO[:, index], f * Data_FNO[:, index]
        Data_T2_PINO, Data_T2_FNO = De1_PINO[:, index], De1_FNO[:, index]
        Data_T3_PINO, Data_T3_FNO = De2_PINO[:, index], De2_FNO[:, index]
        label1 = "Ground truth"
        label2 = "With EN (V2)"
        label3 = "Prediction with PINO as backbone (Case3)"
        label4 = "Prediction with FNO as backbone  (Case1)"
        PicName = 'Fig3a2.jpg'
        ylim = [-4, 4, -0.1, 0.1, -5, 5]
        Title = "Vertical vibration of $1^{st}$ bogie ($Z_{b1}$)"

    if index == 5:
        ylabel1 = "Solution (rad)"
        ylabel2 = "$1^{st}$ derivative (rad/s)"
        ylabel3 = "$2^{st}$ derivative (rad/$s^2$)"
        f = 1
        Data_T1_GT, Data_T1_Pre = f * Data_GT[:, index], f * Data_Predict[:, index]
        Data_T2_GT, Data_T2_Pre = De1_GT[:, index], De1_Predict[:, index]
        Data_T3_GT, Data_T3_Pre = De2_GT[:, index], De2_Predict[:, index]

        Data_T1_PINO, Data_T1_FNO = f * Data_PINO[:, index], f * Data_FNO[:, index]
        Data_T2_PINO, Data_T2_FNO = De1_PINO[:, index], De1_FNO[:, index]
        Data_T3_PINO, Data_T3_FNO = De2_PINO[:, index], De2_FNO[:, index]
        label1 = "Ground truth"
        label2 = "With EN (V2)"
        label3 = "Prediction with PINO as backbone (Case3)"
        label4 = "Prediction with FNO as backbone  (Case1)"
        PicName = 'Fig3a5.jpg'
        ylim = [-0.002, 0.002, -0.1, 0.1, -7, 7]
        Title = "Rotation of the $2^{nd}$ bogie " + r"$\beta_b2$"

    if index == 6:
        ylabel1 = "Solution (mm)"
        ylabel2 = "$1^{st}$ derivative (m/s)"
        ylabel3 = "$2^{st}$ derivative (m/$s^2$)"
        f = 1e3
        Data_T1_GT, Data_T1_Pre = f * Data_GT[:, index], f * Data_Predict[:, index]
        Data_T2_GT, Data_T2_Pre = De1_GT[:, index], De1_Predict[:, index]
        Data_T3_GT, Data_T3_Pre = De2_GT[:, index], De2_Predict[:, index]

        Data_T1_PINO, Data_T1_FNO = f * Data_PINO[:, index], f * Data_FNO[:, index]
        Data_T2_PINO, Data_T2_FNO = De1_PINO[:, index], De1_FNO[:, index]
        Data_T3_PINO, Data_T3_FNO = De2_PINO[:, index], De2_FNO[:, index]
        label1 = "Ground truth"
        label2 = "With EN (V2)"
        label3 = "Prediction with PINO as backbone (Case3)"
        label4 = "Prediction with FNO as backbone  (Case1)"
        PicName = 'Fig3a6.jpg'
        ylim = [-4, 4, -0.2, 0.2, -20, 20]
        Title = "Vertical vibration of the $1^{st}$ wheelset ($Z_{w1}$)"

    if index == 13:
        ylabel1 = "Solution (mm)"
        ylabel2 = "$1^{st}$ derivative (m/s)"
        ylabel3 = "$2^{st}$ derivative (m/$s^2$)"
        f = 1e3
        Data_T1_GT, Data_T1_Pre = f * Data_GT[:, index], f * Data_Predict[:, index]
        Data_T2_GT, Data_T2_Pre = De1_GT[:, index], De1_Predict[:, index]
        Data_T3_GT, Data_T3_Pre = De2_GT[:, index], De2_Predict[:, index]

        Data_T1_PINO, Data_T1_FNO = f * Data_PINO[:, index], f * Data_FNO[:, index]
        Data_T2_PINO, Data_T2_FNO = De1_PINO[:, index], De1_FNO[:, index]
        Data_T3_PINO, Data_T3_FNO = De2_PINO[:, index], De2_FNO[:, index]
        label1 = "Ground truth"
        label2 = "With EN (V2)"
        label3 = "Prediction with PINO as backbone (Case3)"
        label4 = "Prediction with FNO as backbone  (Case1)"
        PicName = 'Fig3a13.jpg'
        ylim = [0.4, 0.6, -0.02, 0.02, -5, 5]
        Title = "Rail deformation under the $4^{th}$ wheelset ($Z_{rw4}$)"

    fontsize = 12
    legend_size = 10
    linewidth1 = 1
    linewidth2 = 0.75
    fig, axs = plt.subplots(3, 1, figsize=(6, 6), dpi=1600)
    fig.suptitle(Title, fontsize=16)
    axs[0].plot(axis1, Data_T1_GT, color='red', linewidth=linewidth1, zorder=4, label=label1)
    axs[0].plot(axis1, Data_T1_Pre, color='blue', linestyle='--', linewidth=linewidth1, zorder=5, label=label2)
    axs[0].plot(axis1, Data_T1_PINO, color='yellow', linestyle=':', linewidth=linewidth1, zorder=3, label=label3)
    axs[0].plot(axis1, Data_T1_FNO, color='grey', linestyle='-', linewidth=linewidth1, zorder=2, label=label4)
    axs[0].set_xlim(0.5, 4.5)
    axs[0].set_ylim(ylim[0], ylim[1])
    axs[0].tick_params(labelsize=12)
    # axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel(ylabel1, fontsize=fontsize)
    if index == 0:
        axs[0].legend(ncol=2, loc='lower right', frameon=False, fontsize=legend_size)

    axs[1].plot(axis2, Data_T2_GT, color='red', linewidth=linewidth1, zorder=4)
    axs[1].plot(axis2, Data_T2_Pre, color='blue', linestyle='--', linewidth=linewidth1, zorder=5)
    axs[1].plot(axis2, Data_T2_PINO, color='yellow', linestyle=':', linewidth=linewidth1, zorder=3)
    axs[1].plot(axis2, Data_T2_FNO, color='grey', linestyle='-', linewidth=linewidth1, zorder=2)
    axs[1].set_xlim(0.5, 4.5)
    axs[1].set_ylim(ylim[2], ylim[3])
    axs[1].tick_params(labelsize=fontsize)
    # axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel(ylabel2, fontsize=fontsize)

    axs[2].plot(axis2, Data_T3_GT, color='red', linewidth=linewidth2, zorder=4)
    axs[2].plot(axis2, Data_T3_Pre, color='blue', linestyle='--', linewidth=linewidth2, zorder=5)
    axs[2].plot(axis2, Data_T3_PINO, color='yellow', linestyle='-', linewidth=0.5, zorder=3)
    axs[2].plot(axis2, Data_T3_FNO, color='grey', linestyle='-', linewidth=0.2, zorder=2)
    axs[2].set_xlim(0.5, 4.5)
    axs[2].set_ylim(ylim[4], ylim[5])
    axs[2].tick_params(labelsize=fontsize)
    axs[2].set_xlabel('Time (s)', fontsize=fontsize)
    axs[2].set_ylabel(ylabel3, fontsize=fontsize)
    fig.tight_layout()
    axs[0].margins(0.0, 0.3)
    axs[1].margins(0.0, 0.3)
    axs[2].margins(0.0, 0.3)
    axs[0].ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    axs[1].ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    axs[2].ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)
    plt.savefig(SavePath + PicName, transparent=False, dpi=1600, bbox_inches='tight')
    # plt.show()




