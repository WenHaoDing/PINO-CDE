import numpy as np
import pylab as plt
import matplotlib as mpl

# plt.rc('font', family='Arial')
# fig = plt.figure(figsize=(1, 2), dpi=200)
# cmap = mpl.cm.jet
# # ax3 = fig.add_axes([0.3, 0.2, 0.2, 0.4])  # 四个参数分别是左、下、宽、长
# ax3 = fig.add_axes([0.3, 0.2, 0.2, 1])  # 四个参数分别是左、下、宽、长
# Value = 1624
# PicName = 'Bar_Acl_MeshA.jpg'
# norm = mpl.colors.Normalize(vmin=0, vmax=Value)
# bounds = [round(elem, 3) for elem in np.linspace(0, Value, 6)]  #
# cb3 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap,
#                                 norm=norm,
#                                 # to use 'extend', you must
#                                 # specify two extra boundaries:
#                                 extend='both',
#                                 ticks=bounds,  # optional
#                                 spacing='proportional',
#                                 orientation='vertical')
# cb3.outline.set_visible(False)
SavePath = 'E:/学习库/文章/PINO-MBD_Physics-Informed Neual Operator for multi-body dynamics/Nature文献/Draft/Figures/ExtendedDataFig1/'
# plt.savefig(SavePath + PicName, transparent=True, dpi=800, bbox_inches='tight')
# plt.show()

plt.rc('font', family='Arial')
plt.figure(figsize=(8,  6.5), dpi=1600)
xticks = np.arange(0.25, 1.1, 0.25)
print(xticks)
yticks = range(3)
# index_ls = ['Acceleration', 'Velocity', 'Displacement']
index_ls = [r'$\mathcal{L}_{veq}$', r'$\mathcal{L}_{veq}+\mathcal{L}_{dde(+)}$', 'Ground truth']
_ = plt.yticks(yticks, index_ls)
_ = plt.xticks(xticks, xticks, fontsize=16)

plt.xlim(0.15, 1.125)
plt.ylim(-0.5, 2.55)
plt.yticks(rotation=45, fontsize=16)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('Time (s)', fontsize=16)
plt.savefig(SavePath + 'Empty frame.jpg', transparent=True, dpi=1600, bbox_inches='tight')
# plt.show()


#
# Prediction,  = plt.plot([0, 1, 2], [0, 1, 2], label="Prediction")
# Truth, = plt.plot([0, 1, 2], [0, 1, 2], label="Ground truth")
# plt.legend(ncol=2)
# plt.savefig(SavePath + '2.jpg', transparent=True, dpi=800, bbox_inches='tight')
# plt.show()