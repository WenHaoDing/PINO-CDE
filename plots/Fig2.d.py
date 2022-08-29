import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Arial')
import h5py
import torch
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import os
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
Path = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/data/Project_HM'
Path += '/Weights_Medium_5000_V3.mat'
LM = torch.tensor(h5py.File(Path)['Weights']).to(torch.float32)[:, :5000].permute([1, 0])
plt.rcParams['xtick.direction'] = 'in'

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(17, 2), dpi=1600)
pix = dict(markerfacecolor='red', marker='+', markeredgecolor='red', markersize=0.15)
bplot = axs.boxplot(LM.numpy(),
                    vert=True,
                    patch_artist=True,
                    flierprops=pix)
plt.yscale("log")
plt.ylim(0.001, 2)
plt.ylabel('Equation losses\n(2% perturbation added)', fontsize=12)
plt.yticks(fontsize=16)
plt.xticks([8, 21, 30, 39], [r'Upper flexible body', r'Lower flexible body', r'Upper steel marbles', r'Lower steel marbles'], fontsize=15)

# axs.set_title('Equation losses (with data losses at 2%)', fontsize=6)
color1 = ['#4DBBD5FF']
color2 = ['#00A087FF']
colorX = ['black']
colorY = ['blue']
colorZ = ['red']
colors = []
for i in range(0, 15):
    colors += color1
for i in range(0, 10):
    colors += color2
for i in range(0, 3):
    colors += colorX
for i in range(0, 3):
    colors += colorY
for i in range(0, 3):
    colors += colorZ
for i in range(0, 3):
    colors += colorX
for i in range(0, 3):
    colors += colorY
for i in range(0, 3):
    colors += colorZ

for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

plt.axvline(x=0.75, ls="--", c=color1[0], linewidth=0.5)
plt.axvline(x=15.25, ls="--", c=color1[0], linewidth=0.5)
plt.axvline(x=15.75, ls="--", c=color2[0], linewidth=0.5)
plt.axvline(x=25.25, ls="--", c=color2[0], linewidth=0.5)
plt.axvline(x=25.75, ls="--", c='#E64B35FF', linewidth=0.5)
plt.axvline(x=34.25, ls="--", c='#E64B35FF', linewidth=0.5)
plt.axvline(x=34.75, ls="--", c='#E64B35FF', linewidth=0.5)
plt.axvline(x=43.25, ls="--", c='#E64B35FF', linewidth=0.5)
SavePath = 'E:/学习库/文章/PINO-MBD_Physics-Informed Neual Operator for multi-body dynamics/Nature文献/Draft/Figures/Fig2/'
PicName = 'Fig2.d.jpg'
plt.savefig(SavePath + PicName, transparent=True, dpi=1600, bbox_inches='tight')

# plt.show()
