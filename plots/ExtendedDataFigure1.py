import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch

from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import os

plt.rc('font', family='Arial')
plt.rcParams['xtick.direction'] = 'in'

Path1 = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/FESRunner/NoData_NoBoundary'
Path2 = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/FESRunner/NoData_WithBoundary'
Path3 = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/FESRunner/Full'

Data1 = np.loadtxt(Path1 + '/LossHistory.txt')
Data2 = np.loadtxt(Path2 + '/LossHistory.txt')
Data3 = np.loadtxt(Path3 + '/LossHistory.txt')

Data1 = 100 * Data1[:299, :]
Data2 = 100 * Data2[:299, :]
Data3 = 100 * Data3[:299, :]
x = np.arange(1, 300, 1)
fig, ax = plt.subplots(figsize=(5.5, 4), dpi=800)
index = 5
label1 = r'$\mathcal{L}_{veq}$'
label2 = r'$\mathcal{L}_{veq}+\mathcal{L}_{dde(+)}$'
label3 = r'$\mathcal{L}_{veq}+\mathcal{L}_{dde}$'
plot1 = ax.plot(x, Data1[:, index], linestyle='-', color='red', zorder=10, label=label1)
plot2 = ax.plot(x, Data2[:, index], linestyle='-', color='blue', zorder=9, label=label2)
plot3 = ax.plot(x, Data3[:, index], linestyle='-', color='black', zorder=8, label=label3)
plt.yscale("log")
ax.set_ylabel('Relative L2 loss (%)', fontsize=12)
ax.set_xlabel('Epoch', fontsize=12)
# ax.set_title(r'Relative L2 losses on solutions', fontsize=12)
ax.set_title(r'Relative L2 losses on $1^{st}$ derivatives', fontsize=12)

# ax.set_xticks(x)
# ax.set_yticks([0.5, 1, 2, 3, 4, 5, 10, 100, 1000])
# ax.set_xticklabels(labels, fontsize=14)
ax.set_ylim(1, 200)
ax.set_xlim(1, 300)
ax.legend(frameon=False, fontsize=12, ncol=1)
plt.tick_params(labelsize=10)
# ax.set_xticklabels(labels, fontsize=14)
SavePath = 'E:/学习库/文章/PINO-MBD_Physics-Informed Neual Operator for multi-body dynamics/Nature文献/Draft/Figures/ExtendedDataFig1/'
PicName = 'ExtendedDataFig2.jpg'
plt.savefig(SavePath + PicName, transparent=True, dpi=800, bbox_inches='tight')
