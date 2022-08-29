"""
Post Processing: Project_Building
This program generates the following figure data:
(1). Time series data for the entire structure's stress information
(Tensor size: nt*ndof*(6*2)*1)
Part of this data will be used for generating stress 3D figures in Matlab with the "patch" function
These data will also be used for stress reliability assessment (also appears as 3D figures in Matlab).
(2). Stress information for structure's local area.
A local position of the structure will be inspected with stress data. These data will be used for
stress heatmap in Origin.
"""
import numpy as np
# import sys
# sys.path.append("..")
# from train_utils.datasets import MatReader
import h5py
import torch
import matplotlib.pyplot as plt
import os
import scipy.io as io
from scipy import stats
import scipy
"""
*******************************************************************PART ONE: Solving and Saving the Maximum Stress Field
"""
Switch = 'Probability Evolution Inspection'

if Switch == 'Eval':
    # Define structure parameters
    E = 3.55e10
    NU = 0.2
    Mode = 200
    # Define FEM mode information
    Path = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/data/Project_BSA/StructureInfo_B.mat'
    W2 = torch.tensor(h5py.File(Path)['W2'][:]).permute([1, 0]).to(torch.float32)
    EBStoreV2 = torch.tensor(h5py.File(Path)['EBStoreV2'][:]).permute([2, 1, 0]).to(torch.float32)
    NodeList = torch.tensor(h5py.File(Path)['NodeList'][:]).permute([1, 0])
    DOF_mapping = torch.tensor(h5py.File(Path)['DOF_mapping'][:]).permute([1, 0]) - 1
    # re-normalization information
    Path = "F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/data/Project_BSA/150.mat"
    StdV, MeanV = torch.tensor(h5py.File(Path)['StdV'][:]).permute([1, 0]).to(torch.float32), torch.tensor(
        h5py.File(Path)['MeanV'][:]).permute([1, 0]).to(torch.float32)
    StdV, MeanV = StdV[:, 7:7 + Mode], MeanV[:, 7:7 + Mode]
    # Start Loop one: Loading different data packs
    for loop in range(0, 11):
        # Define displacement field
        Path = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/BSARunner/Experiment1/eval/EV1'
        FileName = Path + '/eval49999_' + str(loop) + '.pt'
        Dis_PINO = torch.load(FileName)
        eval_dataset = torch.utils.data.TensorDataset(Dis_PINO)
        batchsize = 4
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batchsize, shuffle=False)
        index = 0
        for tem_Dis in eval_loader:
            tem_Dis = tem_Dis[0]
            # Signal re-normalization
            tem_Dis = tem_Dis * StdV + MeanV
            # Mission I : Recover full stress field at multiple time sequence
            Step = 2
            tem_Dis = tem_Dis[:, 150::Step, :]
            NodeDis_PINO = torch.matmul(W2, tem_Dis.unsqueeze(-1)).squeeze(-1)
            u_PINO = NodeDis_PINO[:, :, DOF_mapping.numpy()].unsqueeze(-1)
            stress = 1e-6 * torch.matmul(EBStoreV2[:, :3, :], u_PINO).squeeze(-1)  # We only look at direct stress
            Maximum_Stress = torch.max(torch.max(stress, dim=-1).values, dim=1).values
            # print('Shape of EBStoreV2 is {} and u tensor is {}'.format(EBStoreV2.shape, u_PINO.shape))
            # print('Shape of stress field tensor is {}'.format(stress.shape))
            # print('Shape of maximum stress field tensor is {}'.format(Maximum_Stress.shape))
            if index == 0:
                Maximum_Store = Maximum_Stress
            else:
                Maximum_Store = torch.cat([Maximum_Store, Maximum_Stress], dim=0)
            index += 1
            print('Evaluation process at {}/{} (Loop {}/{})'.format(index, int(Dis_PINO.size(0) / batchsize), loop, 10))
            # print('Maximum stress field shape is now {}'.format(Maximum_Store.size(0)))
        SavePath = Path + '/Stress'
        MaximumFileName = SavePath + 'Maximum_stress_field_Pack' + str(loop) + '.pt'
        torch.save(Maximum_Store, MaximumFileName)
    """
    ***************************************************************PART TWO: Solving the Reliability of The Entire Structure
    """
elif Switch == 'Assemble':
    index = 1
    for loop in range(1, 11):
        SavePath = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/BSARunner/Experiment1/eval/'
        MaximumFileName = SavePath + 'StressMaximum_stress_field_Pack' + str(loop) + '.pt'
        if loop == 1:
            MaxStress = torch.load(MaximumFileName)
        else:
            MaxStress = torch.cat([MaxStress, torch.load(MaximumFileName)], dim=0)
        print('Assemble process at {}/{}'.format(index, 10))
        print(MaxStress.shape)
        index += 1
    mat_path = SavePath + 'MaxStress.mat'
    io.savemat(mat_path, {'MaxStress': MaxStress.numpy()})

elif Switch == 'Probability Evolution Inspection':
    E = 3.55e10
    NU = 0.2
    Mode = 200
    case = 1
    # Define FEM mode information
    Path = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/data/Project_BSA/StructureInfo_B.mat'
    W2 = torch.tensor(h5py.File(Path)['W2'][:]).permute([1, 0]).to(torch.float32)
    EBStoreV2 = torch.tensor(h5py.File(Path)['EBStoreV2'][:]).permute([2, 1, 0]).to(torch.float32)
    NodeList = torch.tensor(h5py.File(Path)['NodeList'][:]).permute([1, 0])
    DOF_mapping = torch.tensor(h5py.File(Path)['DOF_mapping'][:]).permute([1, 0]) - 1
    # re-normalization information
    Path = "F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/data/Project_BSA/150.mat"
    StdV, MeanV = torch.tensor(h5py.File(Path)['StdV'][:]).permute([1, 0]).to(torch.float32), torch.tensor(
        h5py.File(Path)['MeanV'][:]).permute([1, 0]).to(torch.float32)
    StdV, MeanV = StdV[:, 7:7 + Mode], MeanV[:, 7:7 + Mode]

    """
    Generate Stress Inspection Parts first
    Evaluating stress field at all points is memory consuming, we intend to generate probability density evolution figures
    Therefore, we only need to evaluate solution at certain nodes.
    In this case, DOF_mapping only need to contain mapping information for inspected elements (index in Inspection).
    """
    Inspection = np.array([37717, 37634, 31245, 30085, 40374, 1769, 43729])
    Inspection -= 1
    DOF_mapping = DOF_mapping.numpy()
    DOF_mapping = DOF_mapping[Inspection, :]
    DOF_mapping = DOF_mapping.reshape((len(Inspection) * 24))
    W2 = W2[DOF_mapping, :]
    EBStoreV2 = EBStoreV2[Inspection, :, :]
    index = 0
    if case == 0:
        for loop in range(1, 11):
            print('Now opearting loop {}'.format(loop))
            # Define displacement field
            Path = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/BSARunner/Experiment1/eval/EV1'
            FileName = Path + '/eval49999_' + str(loop) + '.pt'
            Dis_PINO = torch.load(FileName)
            eval_dataset = torch.utils.data.TensorDataset(Dis_PINO)
            batchsize = 5000
            eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batchsize, shuffle=False)
            for tem_Dis in eval_loader:
                tem_Dis = tem_Dis[0]
                # Signal re-normalization
                tem_Dis = tem_Dis * StdV + MeanV
                # Mission I : Recover full stress field at multiple time sequence
                Step = 1
                tem_Dis = tem_Dis[:, ::Step, :]
                NodeDis_PINO = torch.matmul(W2, tem_Dis.unsqueeze(-1)).squeeze(-1)
                NodeDis_PINO = NodeDis_PINO.reshape(NodeDis_PINO.size(0), NodeDis_PINO.size(1), len(Inspection), -1).unsqueeze(-1)
                stress = 1e-6 * torch.matmul(EBStoreV2[:, :3, :], NodeDis_PINO).squeeze(-1)  # We only look at direct stress
                stress = torch.max(stress, dim=-1).values
                if index == 0:
                    Statistic = stress
                else:
                    Statistic = torch.cat((Statistic, stress), dim=0)
                    print(Statistic.shape)
                index += 1
        SavePath = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/PDEM/'
        mat_path = SavePath + 'MaxStress_PINO_MBD_49999.mat'
        tensor_path = SavePath + 'MaxStress_PINO_MBD_49999.pt'
        torch.save(Statistic, tensor_path)
    else:
        SavePath = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/PDEM/'
        tensor_path = SavePath + 'MaxStress_PINO_MBD_49999.pt'
        # io.savemat(mat_path, {'Statistic': Statistic.numpy()})
        Statistic = torch.load(tensor_path)
        Statistic = 1e6 * Statistic
    print('The shape of Statistic is {}'.format(Statistic.shape))
    Path = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/PDEM/Span_499V2.mat'
    SpanL = torch.tensor(h5py.File(Path)['SpanL'][:]).permute([1, 0]).to(torch.float32)
    SpanR = torch.tensor(h5py.File(Path)['SpanR'][:]).permute([1, 0]).to(torch.float32)
    Space_DPI = 1500
    RSpan = SpanR - SpanL
    SpanL -= 0.2 * RSpan
    SpanR += 0.2 * RSpan
    print(SpanL.size(1))
    SpaceTick = np.zeros((Space_DPI, SpanL.size(1)))
    for i in range(0, SpanL.size(1)):
        SpaceTick[:, i] = np.linspace(SpanL[0, i], SpanR[0, i], Space_DPI)

    # Statistic = Statistic / 1e6
    # SpaceTick = SpaceTick / 1e6

    PDE_Result = np.zeros((Statistic.size(2), Statistic.size(1), Space_DPI))
    PDE_Stick = np.zeros((Statistic.size(2), Statistic.size(1), Space_DPI))
    for i in range(0, Statistic.size(1)):
        for j in range(0, Statistic.size(2)):
            data = Statistic[:, i, j]
            hist = np.histogram(data, bins=1500)
            hist_dist = scipy.stats.rv_histogram(hist)
            pdf = hist_dist.pdf(SpaceTick[:, j])
            PDE_Result[j, i, :] = pdf
            PDE_Stick[j, i, :] = hist[0]
    io.savemat(SavePath + 'FusedReliability_PINO_MBD_49999.mat', {'PMatrix_Fusion': PDE_Result, 'Check': PDE_Stick})
    plt.plot(PDE_Result[0, 500, :])
    plt.show()






