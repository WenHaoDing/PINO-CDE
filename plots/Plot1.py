import numpy as np
import matplotlib.pyplot as plt

Path1 = 'F:\Pycharm\ExistingPytorch\GNN_Series\Physics-Informed Neural Operator\PINO-Project1\checkpoints\VTCDRunner\V1'
Path2 = 'F:\Pycharm\ExistingPytorch\GNN_Series\Physics-Informed Neural Operator\PINO-Project1\checkpoints\VTCDRunner\V1'

# demonstration of boxplot: https://blog.csdn.net/mighty13/article/details/117402898
# matplotlib color visualization: https://zhuanlan.zhihu.com/p/65220518

Clip = 3
Step = 300

History1 = np.zeros((Step, 3))
History2 = np.zeros((Step, 3))
Store1 = np.zeros((Clip, 3))
Store2 = np.zeros((Clip, 3))
for i in range(0, Clip):
    FileName1 = Path1 + '/eval' + str(i+1) +'.txt'
    # FileName1 = Path1 + '/eval' + str(i+1) +'_GradNorm.txt'
    FileName2 = Path2 + '/eval' + str(i+1) +'.txt'
    # FileName2 = Path1 + '/eval' + str(i+1) +'_NoGradNorm.txt'

    Data1 = np.loadtxt(FileName1)
    Data2 = np.loadtxt(FileName2)
    Store1[i, :] = Data1[Step-2, 4:7]
    Store2[i, :] = Data2[Step-2, 4:7]
    History1 += Data1[:Step, 4:7]
    History2 += Data2[:Step, 4:7]
History1 /= Clip
History2 /= Clip

# Plot No.1 Statistical properties' comparison of two algorithm
plt.figure(1)
labels = 'Solution', '1st Derivative', '2st Derivative'
plot1 = plt.boxplot([Store1[:, 0], Store1[:, 1], Store1[:, 2]], labels=labels, patch_artist=True, whiskerprops={'color': 'k', 'linewidth': 1, 'linestyle': '-'})
plot2 = plt.boxplot([Store2[:, 0], Store2[:, 1], Store2[:, 2]], labels=labels, patch_artist=True,
                    boxprops={'linewidth': 2, 'linestyle': '--'}, whiskerprops={'color': 'b', 'linewidth': 1, 'linestyle': '--'})
colors = ['lightpink', 'lightblue', 'lightgreen']
colors2 = ['lightpink', 'lightblue', 'lightgreen']

for patch, color in zip(plot1['boxes'], colors):
    patch.set_facecolor(color)
for patch, color in zip(plot2['boxes'], colors2):
    patch.set_facecolor(color)
plt.xlabel('Targets')
plt.ylabel('Relative L2 accuracy (%)')
plt.title(label="Comparison of algorithm accuracy",
          fontsize=10,
          color="green")

# Plot No.2 Loss history comparison of two algorithm
plt.figure(2)
plt.semilogy(History1[:, 0], color='black', linewidth=1.5)
plt.semilogy(History1[:, 1], color='blue', linewidth=1.5)
plt.semilogy(History1[:, 2], color='red', linewidth=1.5)
plt.semilogy(History2[:, 0], color='dimgray', linewidth=1, linestyle='--')
plt.semilogy(History2[:, 1], color='cyan', linewidth=1, linestyle='--')
plt.semilogy(History2[:, 2], color='salmon', linewidth=1, linestyle='--')
plt.xlabel('Targets')
plt.ylabel('Relative L2 accuracy (%)')
plt.title(label="Comparison of algorithm loss history",
          fontsize=10,
          color="blue")
# plt.show()
print(np.average(Store1, axis=0), np.mean(Store1))
print(np.average(Store2, axis=0), np.mean(Store2))

