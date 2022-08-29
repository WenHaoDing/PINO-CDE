import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
plt.rc('font', family='Arial')
Path11 = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/HMRunner/T4'
Path12 = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/HMRunner/T1'
Path13 = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/HMRunner/T5'
Path14 = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/HMRunner/T7'

Path21 = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/VTCDRunner/V1'
Path22 = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/VTCDRunner/V2'
Path23 = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/VTCDRunner/V3'
Path24 = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/VTCDRunner/V4'

Path31 = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/BSARunner/R1'
Path32 = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/BSARunner/R2'
Path33 = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/BSARunner/R3'
Path34 = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/BSARunner/R4'
Clip = [5, 5, 5, 4, 2, 4, 5, 4, 3, 3, 3, 5]
Epoch1 = [75, 148, 298]
Epoch2 = [75, 148, 298]
Epoch2_ = [8, 43, 88]
Epoch3 = [125, 248, 498]
index = 0
Store = np.zeros((12, 5, 3, 3))
for path in [Path11, Path12, Path13, Path14, Path21, Path22, Path23, Path24, Path31, Path32, Path33, Path34]:
    for clip in range(1, Clip[index]+1):
        if index < 7:
            Epoch = Epoch1
        elif index == 7:
            Epoch = Epoch2_
        else:
            Epoch = Epoch3
        FileName = path + '/eval' + str(clip) + '.txt'
        Data = np.loadtxt(FileName)
        print(index)
        Store[index, clip-1, :, :] = Data[Epoch, 4:7]
    index += 1
# Store stores all training quality data, Dimension: data index, Clip, Epoch, target(solution, 1st de, 2st de)
Store *= 100
fig, axs = plt.subplots(nrows=3, ncols=3, constrained_layout=False, figsize=(11, 5.5), dpi=800)
# gs = GridSpec(3, 3, figure=fig)
# axs[0] = fig.add_subplot(gs[0, 0])
# axs[1] = fig.add_subplot(gs[0, 1])
# axs[2] = fig.add_subplot(gs[0, 2])
# axs[3] = fig.add_subplot(gs[1, 0])
# axs[4] = fig.add_subplot(gs[1, 1])
# axs[5] = fig.add_subplot(gs[1, 2])
# axs[6] = fig.add_subplot(gs[2, 0])
# axs[7] = fig.add_subplot(gs[2, 1])
# axs[8] = fig.add_subplot(gs[2, 2])

for item in [axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0], axs[1, 1], axs[1, 2], axs[2, 0], axs[2, 1], axs[2, 2]]:
    item.set_xscale("log")
    item.invert_yaxis()
    item.get_yaxis().get_major_formatter().set_scientific(False)


plt.subplots_adjust(wspace=0.075, hspace=None)
label = ['25%', '50%', '100%']
Data = Store[0, :Clip[0], 0, 0]
plot11 = axs[0, 0].boxplot(Store[0, :Clip[0], :, 0], labels=label, patch_artist=True, boxprops={'color': '#E64B35FF','facecolor': '#E64B35FF'}, whiskerprops={'color': '#E64B35FF', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': '#E64B35FF', 'linewidth': 1}, sym='', widths=0.25, zorder=10, medianprops={'color': '#E64B35FF'}, vert=False)
plot12 = axs[0, 0].boxplot(Store[1, :Clip[1], :, 0], labels=label, patch_artist=True, boxprops={'color': '#4DBBD5FF','facecolor': '#4DBBD5FF'}, whiskerprops={'color': '#4DBBD5FF', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': '#4DBBD5FF', 'linewidth': 1}, sym='', widths=0.35, zorder=8, showmeans=False, medianprops={'color': '#4DBBD5FF'}, vert=False)
plot13 = axs[0, 0].boxplot(Store[2, :Clip[2], :, 0], labels=label, patch_artist=True, boxprops={'color': '#3C5488FF','facecolor': '#3C5488FF'}, whiskerprops={'color': '#3C5488FF', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': '#3C5488FF', 'linewidth': 1}, sym='', widths=0.45, zorder=6, showmeans=False, medianprops={'color': '#3C5488FF'}, vert=False)
plot14 = axs[0, 0].boxplot(Store[3, :Clip[3], :, 0], labels=label, patch_artist=True, boxprops={'color': 'lime', 'facecolor': 'lime'}, whiskerprops={'color': 'lime', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': 'lime', 'linewidth': 1}, sym='', widths=0.55, zorder=4, showmeans=False, medianprops={'color': 'lime'}, vert=False)
axs[0, 0].set_xlim(3, 6)
plt.sca(axs[0, 0])
# plt.xticks([3, 4, 5, 6], [str(3), str(4), str(5), str(6)])

plot21 = axs[0, 1].boxplot(Store[0, :Clip[0], :, 1], labels=label, patch_artist=True, boxprops={'color': '#E64B35FF','facecolor': '#E64B35FF'}, whiskerprops={'color': '#E64B35FF', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': '#E64B35FF', 'linewidth': 1}, sym='', widths=0.25, zorder=10, medianprops={'color': '#E64B35FF'}, vert=False)
plot22 = axs[0, 1].boxplot(Store[1, :Clip[1], :, 1], labels=label, patch_artist=True, boxprops={'color': '#4DBBD5FF','facecolor': '#4DBBD5FF'}, whiskerprops={'color': '#4DBBD5FF', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': '#4DBBD5FF', 'linewidth': 1}, sym='', widths=0.35, zorder=8, showmeans=False, medianprops={'color': '#4DBBD5FF'}, vert=False)
plot23 = axs[0, 1].boxplot(Store[2, :Clip[2], :, 1], labels=label, patch_artist=True, boxprops={'color': '#3C5488FF','facecolor': '#3C5488FF'}, whiskerprops={'color': '#3C5488FF', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': '#3C5488FF', 'linewidth': 1}, sym='', widths=0.45, zorder=6, showmeans=False, medianprops={'color': '#3C5488FF'}, vert=False)
plot24 = axs[0, 1].boxplot(Store[3, :Clip[3], :, 1], labels=label, patch_artist=True, boxprops={'color': 'lime', 'facecolor': 'lime'}, whiskerprops={'color': 'lime', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': 'lime', 'linewidth': 1}, sym='', widths=0.55, zorder=4, showmeans=False, medianprops={'color': 'lime'}, vert=False)
axs[0, 1].set_xlim(3.5, 6.5)
plot31 = axs[0, 2].boxplot(Store[0, :Clip[0], :, 2], labels=label, patch_artist=True, boxprops={'color': '#E64B35FF','facecolor': '#E64B35FF'}, whiskerprops={'color': '#E64B35FF', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': '#E64B35FF', 'linewidth': 1}, sym='', widths=0.25, zorder=10, medianprops={'color': '#E64B35FF'}, vert=False)
plot32 = axs[0, 2].boxplot(Store[1, :Clip[1], :, 2], labels=label, patch_artist=True, boxprops={'color': '#4DBBD5FF','facecolor': '#4DBBD5FF'}, whiskerprops={'color': '#4DBBD5FF', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': '#4DBBD5FF', 'linewidth': 1}, sym='', widths=0.35, zorder=8, showmeans=False, medianprops={'color': '#4DBBD5FF'}, vert=False)
plot33 = axs[0, 2].boxplot(Store[2, :Clip[2], :, 2], labels=label, patch_artist=True, boxprops={'color': '#3C5488FF','facecolor': '#3C5488FF'}, whiskerprops={'color': '#3C5488FF', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': '#3C5488FF', 'linewidth': 1}, sym='', widths=0.45, zorder=6, showmeans=False, medianprops={'color': '#3C5488FF'}, vert=False)
plot34 = axs[0, 2].boxplot(Store[3, :Clip[3], :, 2], labels=label, patch_artist=True, boxprops={'color': 'lime', 'facecolor': 'lime'}, whiskerprops={'color': 'lime', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': 'lime', 'linewidth': 1}, sym='', widths=0.55, zorder=4, showmeans=False, medianprops={'color': 'lime'}, vert=False)
axs[0, 2].set_xlim(3, 45)

plot41 = axs[1, 0].boxplot(Store[4, :Clip[4], :, 0], labels=label, patch_artist=True, boxprops={'color': '#E64B35FF','facecolor': '#E64B35FF'}, whiskerprops={'color': '#E64B35FF', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': '#E64B35FF', 'linewidth': 1}, sym='', widths=0.25, zorder=10, medianprops={'color': '#E64B35FF'}, vert=False)
plot42 = axs[1, 0].boxplot(Store[5, :Clip[5], :, 0], labels=label, patch_artist=True, boxprops={'color': '#4DBBD5FF','facecolor': '#4DBBD5FF'}, whiskerprops={'color': '#4DBBD5FF', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': '#4DBBD5FF', 'linewidth': 1}, sym='', widths=0.35, zorder=8, showmeans=False, medianprops={'color': '#4DBBD5FF'}, vert=False)
plot43 = axs[1, 0].boxplot(Store[6, :Clip[6], :, 0], labels=label, patch_artist=True, boxprops={'color': '#3C5488FF','facecolor': '#3C5488FF'}, whiskerprops={'color': '#3C5488FF', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': '#3C5488FF', 'linewidth': 1}, sym='', widths=0.45, zorder=6, showmeans=False, medianprops={'color': '#3C5488FF'}, vert=False)
plot44 = axs[1, 0].boxplot(Store[7, :Clip[7], :, 0], labels=label, patch_artist=True, boxprops={'color': 'lime', 'facecolor': 'lime'}, whiskerprops={'color': 'lime', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': 'lime', 'linewidth': 1}, sym='', widths=0.55, zorder=4, showmeans=False, medianprops={'color': 'lime'}, vert=False)
axs[1, 0].set_xlim(2.5, 20)
plot51 = axs[1, 1].boxplot(Store[4, :Clip[4], :, 1], labels=label, patch_artist=True, boxprops={'color': '#E64B35FF','facecolor': '#E64B35FF'}, whiskerprops={'color': '#E64B35FF', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': '#E64B35FF', 'linewidth': 1}, sym='', widths=0.25, zorder=10, medianprops={'color': '#E64B35FF'}, vert=False)
plot52 = axs[1, 1].boxplot(Store[5, :Clip[5], :, 1], labels=label, patch_artist=True, boxprops={'color': '#4DBBD5FF','facecolor': '#4DBBD5FF'}, whiskerprops={'color': '#4DBBD5FF', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': '#4DBBD5FF', 'linewidth': 1}, sym='', widths=0.35, zorder=8, showmeans=False, medianprops={'color': '#4DBBD5FF'}, vert=False)
plot53 = axs[1, 1].boxplot(Store[6, :Clip[6], :, 1], labels=label, patch_artist=True, boxprops={'color': '#3C5488FF','facecolor': '#3C5488FF'}, whiskerprops={'color': '#3C5488FF', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': '#3C5488FF', 'linewidth': 1}, sym='', widths=0.45, zorder=6, showmeans=False, medianprops={'color': '#3C5488FF'}, vert=False)
plot54 = axs[1, 1].boxplot(Store[7, :Clip[7], :, 1], labels=label, patch_artist=True, boxprops={'color': 'lime', 'facecolor': 'lime'}, whiskerprops={'color': 'lime', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': 'lime', 'linewidth': 1}, sym='', widths=0.55, zorder=4, showmeans=False, medianprops={'color': 'lime'}, vert=False)
axs[1, 1].set_xlim(2, 30)
plot61 = axs[1, 2].boxplot(Store[4, :Clip[4], :, 2], labels=label, patch_artist=True, boxprops={'color': '#E64B35FF','facecolor': '#E64B35FF'}, whiskerprops={'color': '#E64B35FF', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': '#E64B35FF', 'linewidth': 1}, sym='', widths=0.25, zorder=10, medianprops={'color': '#E64B35FF'}, vert=False)
plot62 = axs[1, 2].boxplot(Store[5, :Clip[5], :, 2], labels=label, patch_artist=True, boxprops={'color': '#4DBBD5FF','facecolor': '#4DBBD5FF'}, whiskerprops={'color': '#4DBBD5FF', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': '#4DBBD5FF', 'linewidth': 1}, sym='', widths=0.35, zorder=8, showmeans=False, medianprops={'color': '#4DBBD5FF'}, vert=False)
plot63 = axs[1, 2].boxplot(Store[6, :Clip[6], :, 2], labels=label, patch_artist=True, boxprops={'color': '#3C5488FF','facecolor': '#3C5488FF'}, whiskerprops={'color': '#3C5488FF', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': '#3C5488FF', 'linewidth': 1}, sym='', widths=0.45, zorder=6, showmeans=False, medianprops={'color': '#3C5488FF'}, vert=False)
plot64 = axs[1, 2].boxplot(Store[7, :Clip[7], :, 2], labels=label, patch_artist=True, boxprops={'color': 'lime', 'facecolor': 'lime'}, whiskerprops={'color': 'lime', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': 'lime', 'linewidth': 1}, sym='', widths=0.55, zorder=4, showmeans=False, medianprops={'color': 'lime'}, vert=False)
axs[1, 2].set_xlim(3, 4000)

plot71 = axs[2, 0].boxplot(Store[8, :Clip[8], :, 0], labels=label, patch_artist=True, boxprops={'color': '#E64B35FF','facecolor': '#E64B35FF'}, whiskerprops={'color': '#E64B35FF', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': '#E64B35FF', 'linewidth': 1}, sym='', widths=0.25, zorder=10, medianprops={'color': '#E64B35FF'}, vert=False)
plot72 = axs[2, 0].boxplot(Store[9, :Clip[9], :, 0], labels=label, patch_artist=True, boxprops={'color': '#4DBBD5FF','facecolor': '#4DBBD5FF'}, whiskerprops={'color': '#4DBBD5FF', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': '#4DBBD5FF', 'linewidth': 1}, sym='', widths=0.35, zorder=8, showmeans=False, medianprops={'color': '#4DBBD5FF'}, vert=False)
plot73 = axs[2, 0].boxplot(Store[10, :Clip[10], :, 0], labels=label, patch_artist=True, boxprops={'color': '#3C5488FF','facecolor': '#3C5488FF'}, whiskerprops={'color': '#3C5488FF', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': '#3C5488FF', 'linewidth': 1}, sym='', widths=0.45, zorder=6, showmeans=False, medianprops={'color': '#3C5488FF'}, vert=False)
plot74 = axs[2, 0].boxplot(Store[11, :Clip[11], :, 0], labels=label, patch_artist=True, boxprops={'color': 'lime', 'facecolor': 'lime'}, whiskerprops={'color': 'lime', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': 'lime', 'linewidth': 1}, sym='', widths=0.55, zorder=4, showmeans=False, medianprops={'color': 'lime'}, vert=False)
axs[2, 0].set_xlim(1.5, 25)
plot81 = axs[2, 1].boxplot(Store[8, :Clip[8], :, 1], labels=label, patch_artist=True, boxprops={'color': '#E64B35FF','facecolor': '#E64B35FF'}, whiskerprops={'color': '#E64B35FF', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': '#E64B35FF', 'linewidth': 1}, sym='', widths=0.25, zorder=10, medianprops={'color': '#E64B35FF'}, vert=False)
plot82 = axs[2, 1].boxplot(Store[9, :Clip[9], :, 1], labels=label, patch_artist=True, boxprops={'color': '#4DBBD5FF','facecolor': '#4DBBD5FF'}, whiskerprops={'color': '#4DBBD5FF', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': '#4DBBD5FF', 'linewidth': 1}, sym='', widths=0.35, zorder=8, showmeans=False, medianprops={'color': '#4DBBD5FF'}, vert=False)
plot83 = axs[2, 1].boxplot(Store[10, :Clip[10], :, 1], labels=label, patch_artist=True, boxprops={'color': '#3C5488FF','facecolor': '#3C5488FF'}, whiskerprops={'color': '#3C5488FF', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': '#3C5488FF', 'linewidth': 1}, sym='', widths=0.45, zorder=6, showmeans=False, medianprops={'color': '#3C5488FF'}, vert=False)
plot84 = axs[2, 1].boxplot(Store[11, :Clip[11], :, 1], labels=label, patch_artist=True, boxprops={'color': 'lime', 'facecolor': 'lime'}, whiskerprops={'color': 'lime', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': 'lime', 'linewidth': 1}, sym='', widths=0.55, zorder=4, showmeans=False, medianprops={'color': 'lime'}, vert=False)
axs[2, 1].set_xlim(5, 40)
plot91 = axs[2, 2].boxplot(Store[8, :Clip[8], :, 2], labels=label, patch_artist=True, boxprops={'color': '#E64B35FF','facecolor': '#E64B35FF'}, whiskerprops={'color': '#E64B35FF', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': '#E64B35FF', 'linewidth': 1}, sym='', widths=0.25, zorder=10, medianprops={'color': '#E64B35FF'}, vert=False)
plot92 = axs[2, 2].boxplot(Store[9, :Clip[9], :, 2], labels=label, patch_artist=True, boxprops={'color': '#4DBBD5FF','facecolor': '#4DBBD5FF'}, whiskerprops={'color': '#4DBBD5FF', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': '#4DBBD5FF', 'linewidth': 1}, sym='', widths=0.35, zorder=8, showmeans=False, medianprops={'color': '#4DBBD5FF'}, vert=False)
plot93 = axs[2, 2].boxplot(Store[10, :Clip[10], :, 2], labels=label, patch_artist=True, boxprops={'color': '#3C5488FF','facecolor': '#3C5488FF'}, whiskerprops={'color': '#3C5488FF', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': '#3C5488FF', 'linewidth': 1}, sym='', widths=0.45, zorder=6, showmeans=False, medianprops={'color': '#3C5488FF'}, vert=False)
plot94 = axs[2, 2].boxplot(Store[11, :Clip[11], :, 2], labels=label, patch_artist=True, boxprops={'color': 'lime', 'facecolor': 'lime'}, whiskerprops={'color': 'lime', 'linewidth': 1, 'linestyle': '-'}, capprops={'color': 'lime', 'linewidth': 1}, sym='', widths=0.55, zorder=4, showmeans=False, medianprops={'color': 'lime'}, vert=False)
axs[2, 2].set_xlim(5, 4000)
for item in [axs[0, 1], axs[0, 2], axs[1, 1], axs[1, 2], axs[2, 1], axs[2, 2]]:
    item.set_yticks([])
plt.setp(axs[0, 0], xticks=[3, 4, 5, 6], xticklabels=[str(3), str(4), str(5), str(6)])
plt.setp(axs[0, 1], xticks=[3.5, 4, 5, 6, 6.5], xticklabels=[str(3.5), str(4), str(5), str(6), str(6.5)])
plt.setp(axs[0, 2], xticks=[3, 10, 45], xticklabels=[str(3), str(10), str(45)])
plt.setp(axs[1, 0], xticks=[2.5, 3, 4, 6, 10, 20], xticklabels=[str(2.5), str(3), str(4), str(6), str(10), str(20)])
plt.setp(axs[1, 1], xticks=[2, 3, 4, 6, 10, 20, 30], xticklabels=[str(2), str(3), str(4), str(6), str(10), str(20), str(30)])

plt.setp(axs[2, 0], xticks=[1.5, 2, 3, 4, 6, 10, 15, 25], xticklabels=[str(1.5), str(2), str(3), str(4), str(6), str(10), str(15), str(25)])
plt.setp(axs[2, 1], xticks=[5, 6, 10, 20, 30, 40], xticklabels=[str(5), str(6), str(10), str(20), str(30), str(40)])
axs[0, 0].set_title('Solutions', fontsize=14)
axs[0, 1].set_title('$1^{st}$ derivatives', fontsize=14)
axs[0, 2].set_title('$2^{st}$ derivatives', fontsize=14)

axs[0, 0].set_ylabel('Toy example', fontsize=14)
axs[1, 0].set_ylabel('VTCD', fontsize=14)
axs[2, 0].set_ylabel('4-storey building', fontsize=14)

axs[2, 0].set_xlabel('Relative L2 loss (%)', fontsize=14)
axs[2, 1].set_xlabel('Relative L2 loss (%)', fontsize=14)
axs[2, 2].set_xlabel('Relative L2 loss (%)', fontsize=14)
fig.suptitle("Relative L2 loss performance during training", fontsize=14)
PicName = 'Fig5.jpg'
SavePath = 'E:/学习库/文章/PINO-MBD_Physics-Informed Neual Operator for multi-body dynamics/Nature文献/Draft/Figures/Fig5/'
plt.savefig(SavePath + PicName, transparent=True, dpi=800, bbox_inches='tight')
# plt.show()
