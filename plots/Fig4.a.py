import numpy as np
import pylab as plt
import matplotlib as mpl

SavePath = 'E:/学习库/文章/PINO-MBD_Physics-Informed Neual Operator for multi-body dynamics/Nature文献/Draft/Figures/Fig4/'
plt.rc('font', family='Arial')
plt.figure(figsize=(11,  2.4), dpi=1600)
xticks = np.arange(1.75, 3.25, 0.25)
print(xticks)
yticks = range(2)
index_ls = ['Ground truth', 'Prediction']
_ = plt.yticks(yticks, index_ls)
_ = plt.xticks(xticks, xticks, fontsize=10)

plt.xlim(1.85, 3.15)
plt.ylim(-0.5, 1.35)
plt.yticks(rotation=45, fontsize=10)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('Time (s)', fontsize=10)
plt.savefig(SavePath + 'Empty frame.jpg', transparent=True, dpi=1600, bbox_inches='tight')
# plt.show()