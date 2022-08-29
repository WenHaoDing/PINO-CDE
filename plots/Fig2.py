import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import os
# The codes below will be used to generate parts of figures for Fig.1
plt.rc('font', family='Arial')
Path = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/HMRunner/ForFig'
SavePath = 'E:/学习库/文章/PINO-MBD_Physics-Informed Neual Operator for multi-body dynamics/Nature文献/Draft/Figures/Fig2/'
FileName = Path + '/' + 'Performance525.txt'
Data = torch.tensor(np.loadtxt(FileName))
Data_Predict = Data[:, :43]
Data_GT = Data[:, 43:]
print(Data_Predict.shape, Data_GT.shape)

Pick_Flex = np.array([0, 4, 14, 15, 19, 29])
Pick_Rigid = np.array([30, 31, 32])
Pick = np.array([1, 5, 16, 20, 35, 36])
Path = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/data/Project_HM/5000_V3.mat'
StdV = torch.tensor(h5py.File(Path)['StdV']).to(torch.float32)[10:10+43, :].permute([1, 0])
MeanV = torch.tensor(h5py.File(Path)['MeanV']).to(torch.float32)[10:10+43, :].permute([1, 0])
print(StdV)

dt = 0.001
Data_GT = Data_GT * StdV + MeanV
Data_Predict = Data_Predict * StdV + MeanV
De1_GT = (Data_GT[2:, :] - Data_GT[:-2, :]) / (2 * dt)
De1_Predict = (Data_Predict[2:, :] - Data_Predict[:-2, :]) / (2 * dt)
De2_GT = (Data_GT[2:, :] - 2 * Data_GT[1:-1, :] + Data_GT[:-2, :]) / (dt ** 2)
De2_Predict = (Data_Predict[2:, :] - 2 * Data_Predict[1:-1, :] + Data_Predict[:-2, :]) / (dt ** 2)

axis1 = np.linspace(0, 1, 1000)
axis2 = np.linspace(0, 1, 998)
axis3 = np.linspace(0, 1, 998)

for index in [11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43]:

    fig = plt.figure(figsize=(3.6, 2), dpi=400)
    # fig = plt.figure(1)
    ax_T1 = HostAxes(fig, [0.12, 0.12, 0.7, 0.75])
    # ax_T1 = HostAxes(fig, [0.1, 0.12, 0.3, 0.3])

    # parasite additional axes, share x
    ax_T2 = ParasiteAxes(ax_T1, sharex=ax_T1)
    ax_T3 = ParasiteAxes(ax_T1, sharex=ax_T1)

    # append axes
    ax_T1.parasites.append(ax_T2)
    ax_T1.parasites.append(ax_T3)

    # invisible right axis of ax_T1
    ax_T1.axis['right'].set_visible(False)
    ax_T1.axis['top'].set_visible(False)
    ax_T2.axis['right'].set_visible(True)
    ax_T2.axis['right'].major_ticklabels.set_visible(True)
    ax_T2.axis['right'].label.set_visible(True)

    # ***********************
    if index == 11:
        ylabel1 = 'Mode amplitude No.1'
        ylabel2 = 'Mode amplitude No.5'
        ylabel3 = 'Mode amplitude No.15'
        DataO_GT = Data_GT
        DataO_Pre = Data_Predict
        axis = axis1
        f = 1
        Data_T1_GT, Data_T2_GT, Data_T3_GT = f * DataO_GT[:, 0], f * DataO_GT[:, 4], f * DataO_GT[:, 14]
        Data_T1_Pre, Data_T2_Pre, Data_T3_Pre = f * DataO_Pre[:, 0], f * DataO_Pre[:, 4], f * DataO_Pre[:, 14]
        label1 = "Mode No.1"
        label2 = "Mode No.5"
        label3 = "Mode No.15"
        label4 = "Mode No.1"
        label5 = "Mode No.5"
        label6 = "Mode No.15"
        PicName = 'PartA_De0.jpg'
    elif index == 12:
        ylabel1 = 'Mode amplitude No.1'
        ylabel2 = 'Mode amplitude No.5'
        ylabel3 = 'Mode amplitude No.15'
        DataO_GT = De1_GT
        DataO_Pre = De1_Predict
        axis = axis2
        f = 1
        Data_T1_GT, Data_T2_GT, Data_T3_GT = f * DataO_GT[:, 0], f * DataO_GT[:, 4], f * DataO_GT[:, 14]
        Data_T1_Pre, Data_T2_Pre, Data_T3_Pre = f * DataO_Pre[:, 0], f * DataO_Pre[:, 4], f * DataO_Pre[:, 14]
        label1 = "Mode No.1"
        label2 = "Mode No.5"
        label3 = "Mode No.15"
        label4 = "Mode No.1"
        label5 = "Mode No.5"
        label6 = "Mode No.15"
        PicName = 'PartA_De1.jpg'
    elif index == 13:
        ylabel1 = 'Mode amplitude No.1'
        ylabel2 = 'Mode amplitude No.5'
        ylabel3 = 'Mode amplitude No.15'
        DataO_GT = De2_GT
        DataO_Pre = De2_Predict
        axis = axis2
        f = 1
        Data_T1_GT, Data_T2_GT, Data_T3_GT = f * DataO_GT[:, 0], f * DataO_GT[:, 4], f * DataO_GT[:, 14]
        Data_T1_Pre, Data_T2_Pre, Data_T3_Pre = f * DataO_Pre[:, 0], f * DataO_Pre[:, 4], f * DataO_Pre[:, 14]
        label1 = "Mode No.1"
        label2 = "Mode No.5"
        label3 = "Mode No.15"
        label4 = "Mode No.1"
        label5 = "Mode No.5"
        label6 = "Mode No.15"
        PicName = 'PartA_De2.jpg'
    if index == 21:
        ylabel1 = 'Mode amplitude No.1'
        ylabel2 = 'Mode amplitude No.5'
        ylabel3 = 'Mode amplitude No.10'
        DataO_GT = Data_GT
        DataO_Pre = Data_Predict
        axis = axis1
        f = 1
        Data_T1_GT, Data_T2_GT, Data_T3_GT = f * DataO_GT[:, 15+0], f * DataO_GT[:, 15+4], f * DataO_GT[:, 15+9]
        Data_T1_Pre, Data_T2_Pre, Data_T3_Pre = f * DataO_Pre[:, 15+0], f * DataO_Pre[:, 15+4], f * DataO_Pre[:, 15+9]
        label1 = "Mode No.1"
        label2 = "Mode No.5"
        label3 = "Mode No.10"
        label4 = "Mode No.1"
        label5 = "Mode No.5"
        label6 = "Mode No.10"
        PicName = 'PartB_De0.jpg'
    elif index == 22:
        ylabel1 = 'Mode amplitude No.1'
        ylabel2 = 'Mode amplitude No.5'
        ylabel3 = 'Mode amplitude No.10'
        DataO_GT = De1_GT
        DataO_Pre = De1_Predict
        axis = axis2
        f = 1
        Data_T1_GT, Data_T2_GT, Data_T3_GT = f * DataO_GT[:, 15+0], f * DataO_GT[:, 15+4], f * DataO_GT[:, 15+9]
        Data_T1_Pre, Data_T2_Pre, Data_T3_Pre = f * DataO_Pre[:, 15+0], f * DataO_Pre[:, 15+4], f * DataO_Pre[:, 15+9]
        label1 = "Mode No.1"
        label2 = "Mode No.5"
        label3 = "Mode No.10"
        label4 = "Mode No.1"
        label5 = "Mode No.5"
        label6 = "Mode No.10"
        PicName = 'PartB_De1.jpg'
    elif index == 23:
        ylabel1 = 'Mode amplitude No.1'
        ylabel2 = 'Mode amplitude No.5'
        ylabel3 = 'Mode amplitude No.10'
        DataO_GT = De2_GT
        DataO_Pre = De2_Predict
        axis = axis2
        f = 1
        Data_T1_GT, Data_T2_GT, Data_T3_GT = f * DataO_GT[:, 15+0], f * DataO_GT[:, 15+4], f * DataO_GT[:, 15+9]
        Data_T1_Pre, Data_T2_Pre, Data_T3_Pre = f * DataO_Pre[:, 15+0], f * DataO_Pre[:, 15+4], f * DataO_Pre[:, 15+9]
        label1 = "Mode No.1"
        label2 = "Mode No.5"
        label3 = "Mode No.10"
        label4 = "Mode No.1"
        label5 = "Mode No."
        label6 = "Mode No.10"
        PicName = 'PartB_De2.jpg'
    elif index == 31:
        ylabel1 = 'X direction (mm)'
        ylabel2 = 'Y direction (mm)'
        ylabel3 = 'Z direction (mm)'
        DataO_GT = Data_GT
        DataO_Pre = Data_Predict
        axis = axis1
        f = 1
        Data_T1_GT, Data_T2_GT, Data_T3_GT = f * DataO_GT[:, 25+0], f * DataO_GT[:, 25+3], f * DataO_GT[:, 25+6]
        Data_T1_Pre, Data_T2_Pre, Data_T3_Pre = f * DataO_Pre[:, 25+0], f * DataO_Pre[:, 25+3], f * DataO_Pre[:, 25+6]
        label1 = "X direction"
        label2 = "Y direction"
        label3 = "Z direction"
        label4 = "X direction"
        label5 = "Y direction"
        label6 = "Z direction"
        PicName = 'RigidA_De0.jpg'
    elif index == 32:
        ylabel1 = 'X direction (mm/s)'
        ylabel2 = 'Y direction (mm/s)'
        ylabel3 = 'Z direction (mm/s)'
        DataO_GT = De1_GT
        DataO_Pre = De1_Predict
        axis = axis2
        f = 1
        Data_T1_GT, Data_T2_GT, Data_T3_GT = f * DataO_GT[:, 25+0], f * DataO_GT[:, 25+3], f * DataO_GT[:, 25+6]
        Data_T1_Pre, Data_T2_Pre, Data_T3_Pre = f * DataO_Pre[:, 25+0], f * DataO_Pre[:, 25+3], f * DataO_Pre[:, 25+6]
        label1 = "X direction"
        label2 = "Y direction"
        label3 = "Z direction"
        label4 = "X direction"
        label5 = "Y direction"
        label6 = "Z direction"
        PicName = 'RigidA_De1.jpg'
    elif index == 33:
        ylabel1 = 'X direction (mm/$s^2$)'
        ylabel2 = 'Y direction (mm/$s^2$)'
        ylabel3 = 'Z direction (mm/$s^2$)'
        DataO_GT = De2_GT
        DataO_Pre = De2_Predict
        axis = axis2
        f = 1
        Data_T1_GT, Data_T2_GT, Data_T3_GT = f * DataO_GT[:, 25+0], f * DataO_GT[:, 25+3], f * DataO_GT[:, 25+6]
        Data_T1_Pre, Data_T2_Pre, Data_T3_Pre = f * DataO_Pre[:, 25+0], f * DataO_Pre[:, 25+3], f * DataO_Pre[:, 25+6]
        label1 = "X direction"
        label2 = "Y direction"
        label3 = "Z direction"
        label4 = "X direction"
        label5 = "Y direction"
        label6 = "Z direction"
        PicName = 'RigidA_De2.jpg'
    elif index == 41:
        ylabel1 = 'X direction (mm)'
        ylabel2 = 'Y direction (mm)'
        ylabel3 = 'Z direction (mm)'
        DataO_GT = Data_GT
        DataO_Pre = Data_Predict
        axis = axis1
        f = 1
        Data_T1_GT, Data_T2_GT, Data_T3_GT = f * DataO_GT[:, 34+0], f * DataO_GT[:, 34+3], f * DataO_GT[:, 34+6]
        Data_T1_Pre, Data_T2_Pre, Data_T3_Pre = f * DataO_Pre[:, 34+0], f * DataO_Pre[:, 34+3], f * DataO_Pre[:, 34+6]
        label1 = "X direction"
        label2 = "Y direction"
        label3 = "Z direction"
        label4 = "X direction"
        label5 = "Y direction"
        label6 = "Z direction"
        PicName = 'RigidB_De0.jpg'
    elif index == 42:
        ylabel1 = 'X direction (mm/s)'
        ylabel2 = 'Y direction (mm/s)'
        ylabel3 = 'Z direction (mm/s)'
        DataO_GT = De1_GT
        DataO_Pre = De1_Predict
        axis = axis2
        f = 1
        Data_T1_GT, Data_T2_GT, Data_T3_GT = f * DataO_GT[:, 34+0], f * DataO_GT[:, 34+3], f * DataO_GT[:, 34+6]
        Data_T1_Pre, Data_T2_Pre, Data_T3_Pre = f * DataO_Pre[:, 34+0], f * DataO_Pre[:, 34+3], f * DataO_Pre[:, 34+6]
        label1 = "X direction"
        label2 = "Y direction"
        label3 = "Z direction"
        label4 = "X direction"
        label5 = "Y direction"
        label6 = "Z direction"
        PicName = 'RigidB_De1.jpg'
    elif index == 43:
        ylabel1 = 'X direction (mm/$s^2$)'
        ylabel2 = 'Y direction (mm/$s^2$)'
        ylabel3 = 'Z direction (mm/$s^2$)'
        DataO_GT = De1_GT
        DataO_Pre = De1_Predict
        axis = axis2
        f = 1
        Data_T1_GT, Data_T2_GT, Data_T3_GT = f * DataO_GT[:, 34+0], f * DataO_GT[:, 34+3], f * DataO_GT[:, 34+6]
        Data_T1_Pre, Data_T2_Pre, Data_T3_Pre = f * DataO_Pre[:, 34+0], f * DataO_Pre[:, 34+3], f * DataO_Pre[:, 34+6]
        label1 = "X direction"
        label2 = "Y direction"
        label3 = "Z direction"
        label4 = "X direction"
        label5 = "Y direction"
        label6 = "Z direction"
        PicName = 'RigidB_De2.jpg'
    # ***********************
    # set label for axis
    ax_T1.set_ylabel(ylabel1)
    ax_T1.set_xlabel('Time (s)')
    ax_T2.set_ylabel(ylabel2)
    ax_T3.set_ylabel(ylabel3)

    T3_axisline = ax_T3.get_grid_helper().new_fixed_axis

    ax_T3.axis['right2'] = T3_axisline(loc='right', axes=ax_T3, offset=(45, 0))

    fig.add_axes(ax_T1)

    ''' #set limit of x, y
    ax_T1.set_xlim(0,2)
    ax_T1.set_ylim(0,3)
    '''

    color1_GT = '#7E2F8E'
    color1_Pre = '#EDB120'
    color2_GT = '#A2142F'
    color2_Pre = '#4DBEEE'
    color3_GT = 'red'
    color3_Pre = 'blue'

    curve_T1_GT, = ax_T1.plot(axis, Data_T1_GT.numpy(), label=label4, color=color1_GT, zorder=5)
    curve_T1_Pre, = ax_T1.plot(axis, Data_T1_Pre.numpy(), label=label1, linestyle='--', color=color1_Pre, zorder=6)

    curve_T2_GT, = ax_T2.plot(axis, Data_T2_GT.numpy(), label=label5, color=color2_GT, zorder=3)
    curve_T2_Pre, = ax_T2.plot(axis, Data_T2_Pre.numpy(), label=label2, linestyle='--', color=color2_Pre, zorder=4)

    curve_T3_GT, = ax_T3.plot(axis, Data_T3_GT.numpy(), label=label6, color=color3_GT, zorder=1)
    curve_T3_Pre, = ax_T3.plot(axis, Data_T3_Pre.numpy(), label=label3, linestyle='--', color=color3_Pre, zorder=2)


    # ax_T2.set_ylim(0,4)
    # ax_T3.set_ylim(0,4)
    # ax_cp.set_ylim(0,50)
    # ax_wear.set_ylim(0,30)

    # l1 = ax_T1.legend(handles=[curve_T1_Pre, curve_T2_Pre, curve_T3_Pre], loc='lower right', frameon=False, fontsize='small')
    # l2 = ax_T2.legend(handles=[curve_T1_GT, curve_T2_GT, curve_T3_GT], loc='lower center', frameon=False, fontsize='small')
    l1 = ax_T1.legend(handles=[curve_T1_Pre, curve_T2_Pre, curve_T3_Pre], loc='lower right', frameon=False, fontsize=8, ncol=1)

    ax_T1.axis['left'].label.set_color(color1_Pre)
    ax_T2.axis['right'].label.set_color(color2_Pre)
    ax_T3.axis['right2'].label.set_color(color3_Pre)

    ax_T1.axis['left'].major_ticks.set_color(color1_Pre)
    ax_T2.axis['right'].major_ticks.set_color(color2_Pre)
    ax_T3.axis['right2'].major_ticks.set_color(color3_Pre)

    ax_T1.axis['left'].major_ticklabels.set_color(color1_Pre)
    ax_T2.axis['right'].major_ticklabels.set_color(color2_Pre)
    ax_T3.axis['right2'].major_ticklabels.set_color(color3_Pre)

    ax_T1.axis['left'].line.set_color(color1_Pre)
    ax_T2.axis['right'].line.set_color(color2_Pre)
    ax_T3.axis['right2'].line.set_color(color3_Pre)


    # ax_T1.margins(0.0, 0.3)
    # ax_T2.margins(0.0, 0.3)
    # ax_T3.margins(0.0, 0.3)

    ax_T1.set_ylim(2.2 * np.min(Data_T1_GT.numpy()), 1.2 * np.max(Data_T1_GT.numpy()))
    ax_T2.set_ylim(2.2 * np.min(Data_T2_GT.numpy()), 1.2 * np.max(Data_T2_GT.numpy()))
    ax_T3.set_ylim(2.2 * np.min(Data_T3_GT.numpy()), 1.2 * np.max(Data_T3_GT.numpy()))
    ax_T1.set_xlim(0, 1)
    ax_T2.set_xlim(0, 1)
    ax_T3.set_xlim(0, 1)

    ax_T1.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    ax_T2.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    ax_T3.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

    # Change here
    plt.savefig(SavePath + PicName, transparent=False, dpi=1600, bbox_inches='tight')
    # plt.show()
