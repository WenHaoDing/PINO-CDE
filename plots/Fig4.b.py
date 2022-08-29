import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import os
import matplotlib as mpl

plt.rc('font', family='Arial')
Path = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/PDEM/PostProcessedProbability.mat'
PTensor_PINOMBD = torch.tensor(h5py.File(Path)['PMatrix_Fusion']).to(torch.float32)[:, :, :].permute([2, 1, 0])
PTensor_PDEM = torch.tensor(h5py.File(Path)['PMatrix_Fusion_PDEM']).to(torch.float32)[:, :, :].permute([2, 1, 0])
SpaceTicks = torch.tensor(h5py.File(Path)['SpaceTick']).to(torch.float32)[:, :].permute([1, 0]).numpy()
print(PTensor_PINOMBD.shape, SpaceTicks.shape)
SavePath = 'E:/学习库/文章/PINO-MBD_Physics-Informed Neual Operator for multi-body dynamics/Nature文献/Draft/Figures/Fig4/'
# Draw Heatmaps first
switch_c = 1
switch_d = 1
switch_b = 1
if switch_c == 1:
    for Index in [0, 4, 6]:
        y1 = np.linspace(0.2, 5, PTensor_PINOMBD.size(1))
        y2 = np.linspace(0.2, 5, PTensor_PDEM.size(1))
        x = SpaceTicks[:, Index]
        xtick1, ytick1 = np.meshgrid(x, y1)
        xtick2, ytick2 = np.meshgrid(x, y2)
        Data1 = PTensor_PINOMBD[Index, :, :].numpy()
        Data2 = PTensor_PDEM[Index, :, :].numpy()

        if Index == 0:
            ylim = [0.2, 5]
            xlim = [-1e6, 3e6]
            xlabel = 'Main stress (Pa)'
            ylabel = 'Time (s)'
            title1 = 'PINO-MBD'
            title2 = 'PDEM'
            lim = np.linspace(0, 0.2e-5, 20)
            tick = np.around(np.linspace(lim[0], lim[-1], 8), decimals=8)
            Suptitle = 'PDF evolution of main stress at inspection point No.1'
            Name = 'InspectionNo1.jpg'
            t1 = 0
            t2 = 30
            title3 = 'PDF at T=0.2s'
            title4 = 'PDF at T=0.5s'
            slice = [0.2, 0.5]

        elif Index == 1:
            ylim = [0.2, 5]
            xlim = [-1e6, 3e6]
            xlabel = 'Main stress (Pa)'
            ylabel = 'Time (s)'
            title1 = 'PINO-MBD'
            title2 = 'PDEM'
            lim = np.linspace(0, 0.2e-5, 20)
            tick = np.around(np.linspace(lim[0], lim[-1], 8), decimals=8)
            Suptitle = 'PDF evolution of main stress at inspection point No.2'
            Name = 'InspectionNo2.jpg'

        elif Index == 4:
            ylim = [0.2, 5]
            xlim = [-1e6, 3e6]
            xlabel = 'Main stress (Pa)'
            ylabel = 'Time (s)'
            title1 = 'PINO-MBD'
            title2 = 'PDEM'
            lim = np.linspace(0, 0.2e-5, 20)
            tick = np.around(np.linspace(lim[0], lim[-1], 8), decimals=8)
            Suptitle = 'PDF evolution of main stress at inspection point No.2'
            Name = 'InspectionNo5.jpg'
            slice = [1.0, 2.0]

        elif Index == 6:
            ylim = [0.2, 5]
            xlim = [-0.5e6, 2.75e6]
            xlabel = 'Main stress (Pa)'
            ylabel = 'Time (s)'
            title1 = 'PINO-MBD'
            title2 = 'PDEM'
            lim = np.linspace(0, 0.2e-5, 20)
            tick = np.around(np.linspace(lim[0], lim[-1], 8), decimals=8)
            Suptitle = 'PDF evolution of main stress at inspection point No.3'
            Name = 'InspectionNo7.jpg'
            slice = [3.0, 4.0]

        font0 = {'family': 'Arial', 'weight': 'normal', 'style': 'normal', 'size': 15}
        font1 = {'family': 'Arial', 'weight': 'bold', 'style': 'normal', 'size': 15}
        fig1, axs = plt.subplots(1, 2, figsize=(6, 5), dpi=1600)
        c1 = axs[0].contourf(xtick1, ytick1, Data1, levels=lim, cmap=plt.cm.jet)
        c2 = axs[1].contourf(xtick2, ytick2, Data2, levels=lim, cmap=plt.cm.jet)
        axs[0].set_ylim(ylim[0], ylim[1])
        axs[0].set_xlim(xlim[0], xlim[1])
        axs[0].set_xlabel(xlabel, font0)
        axs[0].set_ylabel(ylabel, font0)
        axs[0].set_title(title1, font0)
        axs[1].set_ylim(ylim[0], ylim[1])
        axs[1].set_xlim(xlim[0], xlim[1])
        axs[1].set_xlabel(xlabel, font0)
        axs[1].set_title(title2, font0)
        fig1.subplots_adjust(right=0.9)
        axs[0].axhline(slice[0], color='yellow', linestyle='--', linewidth=1, zorder=10)
        axs[0].axhline(slice[1], color='yellow', linestyle='--', linewidth=1, zorder=10)
        axs[1].axhline(slice[0], color='yellow', linestyle='--', linewidth=1, zorder=10)
        axs[1].axhline(slice[1], color='yellow', linestyle='--', linewidth=1, zorder=10)
        rect = [0.92, 0.12, 0.015, 1-2*0.12]
        if Index == 6:
            cbar_ax = fig1.add_axes(rect)
            cb = plt.colorbar(c2, cax=cbar_ax, label='PDF')
            cb.set_ticks(tick)
            cb.update_ticks()
        fig1.suptitle(Suptitle, fontproperties=font1)
        Name = SavePath + Name
        plt.savefig(Name, transparent=True, dpi=1600, bbox_inches='tight')

if switch_d == 1:
    for Index in [0, 4, 6]:
        y1 = np.linspace(0.2, 5, PTensor_PINOMBD.size(1))
        y2 = np.linspace(0.2, 5, PTensor_PDEM.size(1))
        x = SpaceTicks[:, Index]
        xtick1, ytick1 = np.meshgrid(x, y1)
        xtick2, ytick2 = np.meshgrid(x, y2)
        Data1 = PTensor_PINOMBD[Index, :, :].numpy()
        Data2 = PTensor_PDEM[Index, :, :].numpy()

        if Index == 0:
            zlim = [1.0e-5, 1.5e-5]
            xlim = [-0.25e6, 1e6]
            t1 = 0
            t2 = 61
            xlabel = 'Main stress (Pa)'
            title1 = 'T=0.2s'
            title2 = 'T=0.5s'
            Name = SavePath + '2DNo1T1.jpg'
            sub_xlim = [0.6e6, 1.0e6, 0.2e6, 0.6e6]
            sub_ylim = [0.75e-6, 1e-6]
        elif Index == 4:
            zlim = [0.6e-5, 0.3e-5]
            xlim = [-1e6, 2.5e6]
            t1 = 161
            t2 = 361
            xlabel = 'Main stress (Pa)'
            title1 = 'T=1.0s'
            title2 = 'T=2.0s'
            Name = SavePath + '2DNo2T1.jpg'
            sub_xlim = [0.5e6, 1.5e6, 1.5e6, 2.5e6]
            sub_ylim = [0.25e-6, 0.25e-6]
        elif Index == 6:
            zlim = [0.8e-5, 0.8e-5]
            xlim = [-1e6, 2e6]
            t1 = 561
            t2 = 761
            xlabel = 'Main stress (Pa)'
            title1 = 'T=3.0s'
            title2 = 'T=4.0s'
            Name = SavePath + '2DNo3T1.jpg'
            sub_xlim = [1e6, 2e6, 1e6, 2e6]
            sub_ylim = [0.5e-6, 0.5e-6]

        font0 = {'family': 'Arial', 'weight': 'normal', 'style': 'normal', 'size': 15}
        font1 = {'family': 'Arial', 'weight': 'bold', 'style': 'normal', 'size': 15}
        fig2, axs2 = plt.subplots(2, 1, figsize=(5, 6.35), dpi=1600)
        p1, = axs2[0].plot(x, Data1[t1, :], linestyle='-', color='blue', linewidth=1, zorder=1, label='PINO-MBD')
        p2, = axs2[0].plot(x, Data2[t1, :], color='red', linewidth=1, zorder=3, label='PDEM')
        # plt.legend([p1, p2], ['PINO-MBD', 'PDEM'], loc='upper right', frameon=False, fontsize='small')
        p3, = axs2[1].plot(x, Data1[t2, :], linestyle='-', color='blue', linewidth=1, zorder=1, label='PINO-MBD')
        p4, = axs2[1].plot(x, Data2[t2, :], color='red', linewidth=1, zorder=3, label='PDEM')
        lines, labels = fig2.axes[0].get_legend_handles_labels()
        fig2.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.125, 0.89), frameon=False, fontsize=13)
        # plt.legend([p3, p4], ['PINO-MBD', 'PDEM'], loc='upper right', frameon=False, fontsize=8)
        axs2[0].set_ylim(0, zlim[0])
        axs2[0].set_xlim(xlim[0], xlim[1])
        # axs2[0].set_xlabel(xlabel, font0)
        axs2[0].set_ylabel('PDF', font0)
        axs2[0].set_title(title1, font1)
        axs2[1].set_ylim(0, zlim[1])
        axs2[1].set_xlim(xlim[0], xlim[1])
        axs2[1].set_xlabel(xlabel, font0)
        axs2[1].set_ylabel('PDF', font0)
        axs2[1].set_title(title2, font1)



        ax3 = fig2.add_axes([0.57, 0.65, 0.3, 0.2], zorder=10)
        ax3.ticklabel_format(style='sci', scilimits=(-1, 2), axis='x')
        ax3.plot(x, Data1[t1, :], linestyle='-', color='blue', linewidth=1, zorder=1)
        ax3.plot(x, Data2[t1, :], linestyle='-', color='red', linewidth=1, zorder=3)
        ax3.set_xlim(sub_xlim[0], sub_xlim[1])
        ax3.set_ylim(0, sub_ylim[0])
        ax4 = fig2.add_axes([0.57, 0.227, 0.3, 0.2], zorder=10)
        ax4.ticklabel_format(style='sci', scilimits=(-1, 2), axis='x')
        ax4.plot(x, Data1[t2, :], linestyle='-', color='blue', linewidth=1, zorder=1)
        ax4.plot(x, Data2[t2, :], linestyle='-', color='red', linewidth=1, zorder=3)
        ax4.set_xlim(sub_xlim[2], sub_xlim[3])
        ax4.set_ylim(0, sub_ylim[1])
        ax3.axvline(x=1.71e6, color='black', linestyle='--', linewidth=1, zorder=10)
        ax4.axvline(x=1.71e6, color='black', linestyle='--', linewidth=1, zorder=10)
        plt.savefig(Name, transparent=True, dpi=1600, bbox_inches='tight')

fig = plt.figure(figsize=(1, 2), dpi=200)
cmap = mpl.cm.jet
# ax3 = fig.add_axes([0.3, 0.2, 0.2, 0.4])  # 四个参数分别是左、下、宽、长
ax3 = fig.add_axes([0.3, 0.8, 0.2, 1])  # 四个参数分别是左、下、宽、长
Value = 30
PicName = 'Bar_DamagePro.jpg'
norm = mpl.colors.Normalize(vmin=0, vmax=Value)
bounds = [round(elem, 3) for elem in np.linspace(0, Value, 5)]  #
cb3 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap,
                                norm=norm,
                                # to use 'extend', you must
                                # specify two extra boundaries:
                                extend='both',
                                ticks=bounds,  # optional
                                spacing='proportional',
                                orientation='vertical',
                                label='Mean maximum main stress (%)')
cb3.outline.set_visible(False)
SavePath = 'E:/学习库/文章/PINO-MBD_Physics-Informed Neual Operator for multi-body dynamics/Nature文献/Draft/Figures/Fig4/'
plt.savefig(SavePath + PicName, transparent=True, dpi=800, bbox_inches='tight')
# plt.show()
