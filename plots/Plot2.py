import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import matplotlib.pyplot as plt
# The codes below will be used to generate parts of figures for Fig.1
Path = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/HMRunner/T4'
SavePath = 'E:/学习库/文章/PINO-MBD_Physics-Informed Neual Operator for multi-body dynamics/Nature文献/Draft/Figures/Parts/'
FileName = Path + '/' + 'Performance299.txt'
Data = torch.tensor(np.loadtxt(FileName))
Data_Predict = Data[:, :43]
Data_GT = Data[:, 43:]
print(Data_Predict.shape, Data_GT.shape)

Pick_Flex = np.array([1, 5, 16, 20])
Pick_Rigid = np.array([35, 36])
Pick = np.array([1, 5, 16, 20, 35, 36])
Path = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/data/Project_HM/500_V3.mat'
StdV = torch.tensor(h5py.File(Path)['StdV']).to(torch.float32)[10:10+43, :].permute([1, 0])
MeanV = torch.tensor(h5py.File(Path)['MeanV']).to(torch.float32)[10:10+43, :].permute([1, 0])

dt = 0.001
Data_GT = Data_GT * StdV + MeanV
Data_Predict = Data_Predict * StdV + MeanV
De1_GT = (Data_GT[2:, :] - Data_GT[:-2, :]) / (2 * dt)
De1_Predict = (Data_Predict[2:, :] - Data_Predict[:-2, :]) / (2 * dt)
De2_GT = (Data_GT[2:, :] - 2 * Data_GT[1:-1, :] - Data_GT[:-2, :]) / (dt ** 2)
De2_Predict = (Data_Predict[2:, :] - 2 * Data_Predict[1:-1, :] - Data_Predict[:-2, :]) / (dt ** 2)

axis1 = np.linspace(0, 1, 1000)
axis2 = np.linspace(0, 1, 998)
axis3 = np.linspace(0, 1, 998)
#  Plot for parts
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18}
for i in Pick_Rigid:
    plt.figure(i, figsize=(10, 1))
    plt.plot(Data_Predict[:, i-1].numpy(), color='red', label='Prediction')
    plt.plot(Data_GT[:, i-1].numpy(), linestyle='--', color='blue', label='Ground truth')
    plt.legend(bbox_to_anchor=(-0.165, 1.25), loc='upper left', borderaxespad=1, prop=font1)
    plt.axis('off')
    plt.savefig(SavePath + 'ODE_Solution' + str(i) + '_Compare.png', transparent=True, dpi=800, bbox_inches='tight')
    plt.close()

    plt.figure(i, figsize=(10, 1))
    plt.plot(De1_Predict[:, i-1].numpy(), color='red', label='Prediction')
    plt.plot(De1_GT[:, i-1].numpy(), linestyle='--', color='blue', label='Ground truth')
    plt.legend(bbox_to_anchor=(-0.165, 1.25), loc='upper left', borderaxespad=1, prop=font1)
    plt.axis('off')
    plt.savefig(SavePath + 'ODE_1stDerivative' + str(i) + '_Compare.png', transparent=True, dpi=800, bbox_inches='tight')
    plt.close()

    plt.figure(i, figsize=(10, 1))
    plt.plot(De2_Predict[:, i-1].numpy(), color='red', label='Prediction')
    plt.plot(De2_GT[:, i-1].numpy(), linestyle='--', color='blue', label='Ground truth')
    plt.legend(bbox_to_anchor=(-0.165, 1.25), loc='upper left', borderaxespad=1, prop=font1)
    plt.axis('off')
    plt.savefig(SavePath + 'ODE_2stDerivative' + str(i) + '_Compare.png', transparent=True, dpi=800, bbox_inches='tight')
    plt.close()

for i in Pick:
    plt.figure(i, figsize=(8, 1))
    # ax = plt.axes()
    # ax.set_facecolor('#4DBBD5FF')
    plt.plot(axis1, Data_Predict[:, i-1].numpy(), color='black', label='Prediction')
    plt.axis('off')
    plt.savefig(SavePath + 'FlexSolution' + str(i) + '.png', transparent=True, dpi=800, bbox_inches='tight')
    plt.tight_layout()
    plt.close()


    plt.figure(i, figsize=(8, 1))
    plt.plot(axis2, De1_Predict[:, i-1].numpy(), color='black', label='Prediction')
    plt.axis('off')
    plt.savefig(SavePath + 'Flex1stDerivative' + str(i) + '.png', transparent=True, dpi=800, bbox_inches='tight')
    plt.tight_layout()
    plt.close()

    plt.figure(i, figsize=(8, 1))
    plt.plot(axis2, De2_Predict[:, i-1].numpy(), color='black', label='Prediction')
    plt.axis('off')
    plt.savefig(SavePath + 'Flex2stDerivative' + str(i) + '.png', transparent=True, dpi=800, bbox_inches='tight')
    plt.tight_layout()
    plt.close()








