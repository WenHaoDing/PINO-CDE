import matplotlib.pyplot as plt
import numpy as np
SavePath = 'E:/学习库/文章/PINO-MBD_Physics-Informed Neual Operator for multi-body dynamics/Nature文献/Draft/Figures/Parts/'

y = np.array([15, 15, 15])
colors = ['#4DBBD5FF', '#00A087FF', '#E64B35FF']

plt.pie(y, colors=colors)
plt.savefig(SavePath + 'Color.png', transparent=True, dpi=300, bbox_inches='tight')

plt.show()