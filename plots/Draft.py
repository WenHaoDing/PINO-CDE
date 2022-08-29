import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Arial')
import h5py
import torch
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
# The codes below will be used to generate parts of figures for Fig.1
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

fig = plt.figure(1) #定义figure，（1）中的1是什么
ax_cof = HostAxes(fig, [0.1, 0.12, 0.5, 0.75])  #用[left, bottom, weight, height]的方式定义axes，0 <= l,b,w,h <= 1

#parasite addtional axes, share x
ax_temp = ParasiteAxes(ax_cof, sharex=ax_cof)
ax_load = ParasiteAxes(ax_cof, sharex=ax_cof)
ax_cp = ParasiteAxes(ax_cof, sharex=ax_cof)
ax_wear = ParasiteAxes(ax_cof, sharex=ax_cof)

#append axes
ax_cof.parasites.append(ax_temp)
ax_cof.parasites.append(ax_load)
ax_cof.parasites.append(ax_cp)
ax_cof.parasites.append(ax_wear)

#invisible right axis of ax_cof
ax_cof.axis['right'].set_visible(False)
ax_cof.axis['top'].set_visible(False)
ax_temp.axis['right'].set_visible(True)
ax_temp.axis['right'].major_ticklabels.set_visible(True)
ax_temp.axis['right'].label.set_visible(True)

#set label for axis
ax_cof.set_ylabel('cof')
ax_cof.set_xlabel('Distance (m)')
ax_temp.set_ylabel('Temperature')
ax_load.set_ylabel('load')
ax_cp.set_ylabel('CP')
ax_wear.set_ylabel('Wear')

load_axisline = ax_load.get_grid_helper().new_fixed_axis
cp_axisline = ax_cp.get_grid_helper().new_fixed_axis
wear_axisline = ax_wear.get_grid_helper().new_fixed_axis

ax_load.axis['right2'] = load_axisline(loc='right', axes=ax_load, offset=(40,0))
ax_cp.axis['right3'] = cp_axisline(loc='right', axes=ax_cp, offset=(80,0))
ax_wear.axis['right4'] = wear_axisline(loc='right', axes=ax_wear, offset=(120,0))

fig.add_axes(ax_cof)

''' #set limit of x, y
ax_cof.set_xlim(0,2)
ax_cof.set_ylim(0,3)
'''
Data_T1_GT, Data_T2_GT, Data_T3_GT = Data_GT[:, 0], Data_GT[:, 4], Data_GT[:, 14]
Data_T1_Pre, Data_T2_Pre, Data_T3_Pre = Data_Predict[:, 0], Data_Predict[:, 4], Data_Predict[:, 14]
axis = axis1

curve_cof, = ax_cof.plot(axis, Data_T1_GT.numpy(), label="CoF", color='black')
curve_temp, = ax_temp.plot(axis, Data_T1_Pre.numpy(), label="Temp", color='red')
curve_load, = ax_load.plot(axis, Data_T2_GT.numpy(), label="Load", color='green')
curve_cp, = ax_cp.plot(axis, Data_T2_Pre.numpy(), label="CP", color='pink')
curve_wear, = ax_wear.plot(axis, Data_T3_GT.numpy(), label="Wear", color='blue')

# ax_temp.set_ylim(0,4)
# ax_load.set_ylim(0,4)
# ax_cp.set_ylim(0,50)
# ax_wear.set_ylim(0,30)

ax_temp.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
ax_load.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
ax_cp.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
ax_wear.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')

ax_cof.legend()

#轴名称，刻度值的颜色
#ax_cof.axis['left'].label.set_color(ax_cof.get_color())
ax_temp.axis['right'].label.set_color('red')
ax_load.axis['right2'].label.set_color('green')
ax_cp.axis['right3'].label.set_color('pink')
ax_wear.axis['right4'].label.set_color('blue')

ax_temp.axis['right'].major_ticks.set_color('red')
ax_load.axis['right2'].major_ticks.set_color('green')
ax_cp.axis['right3'].major_ticks.set_color('pink')
ax_wear.axis['right4'].major_ticks.set_color('blue')

ax_temp.axis['right'].major_ticklabels.set_color('red')
ax_load.axis['right2'].major_ticklabels.set_color('green')
ax_cp.axis['right3'].major_ticklabels.set_color('pink')
ax_wear.axis['right4'].major_ticklabels.set_color('blue')

ax_temp.axis['right'].line.set_color('red')
ax_load.axis['right2'].line.set_color('green')
ax_cp.axis['right3'].line.set_color('pink')
ax_wear.axis['right4'].line.set_color('blue')

plt.show()