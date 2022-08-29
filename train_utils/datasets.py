import scipy.io
import numpy as np
import os
import h5py

try:
    from pyDOE import lhs
    # Only needed for PINN's dataset
except ImportError:
    lhs = None

import torch
from torch.utils.data import Dataset
from .utils import get_grid3d, convert_ic, torch2dgrid


def online_loader(sampler, S, T, time_scale, batchsize=1):
    while True:
        u0 = sampler.sample(batchsize)
        a = convert_ic(u0, batchsize,
                       S, T,
                       time_scale=time_scale)
        yield a


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        # self.data = scipy.io.loadmat(self.file_path)
        self.data = h5py.File(self.file_path)
        self.old_mat = True

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field][:]
        # print(x.shape)
        # os.system('pause')
        # x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


class FES_Loader(object):
    def __init__(self, datapath, weights_datapath, datapath_test, weights_datapath_test, Structure_datapath, nt, nSlice,
                 sub_t, new, inputDim, outputDim):
        dataloader = MatReader(datapath)
        dataloader_test = MatReader(datapath_test)
        dataloader_weight = MatReader(weights_datapath)
        dataloader_weight_test = MatReader(weights_datapath_test)
        dataloader_structure = MatReader(Structure_datapath)
        self.sub_t = sub_t
        self.T = nt // sub_t
        self.new = new
        self.nSlice = nSlice
        if new:
            self.T += 1
        self.x_data = dataloader.read_field('input').permute([2, 1, 0])[:, :nt, :]
        self.y_data = dataloader.read_field('output').permute([2, 1, 0])[:, :nt, :]
        self.x_data_test = dataloader_test.read_field('input').permute([2, 1, 0])[:, :nt, :]
        self.y_data_test = dataloader_test.read_field('output').permute([2, 1, 0])[:, :nt, :]
        self.weights_data = dataloader_weight.read_field('Weights').permute([1, 0])
        self.weights_data_test = dataloader_weight_test.read_field('Weights').permute([1, 0])
        self.W2 = dataloader_structure.read_field('W2').permute([1, 0]).to(torch.device('cuda:0'))
        self.Eigens2 = dataloader_structure.read_field('Eigens2').permute([1, 0]).to(torch.device('cuda:0'))
        self.TrackDOFs = dataloader_structure.read_field('TrackDOFs').permute([1, 0])
        self.Nloc = dataloader_structure.read_field('Nloc').permute([1, 0]).int().to(torch.device('cuda:0'))
        print('Structure Information Warning:W2_{};Eigens2_{};TrackDOFs_{};Nloc_{}'.format(self.W2.shape,
                                                                                           self.Eigens2.shape,
                                                                                           self.TrackDOFs.shape,
                                                                                           self.Nloc.shape))
        # Providing two different data ToOne action options
        self.MeanV = dataloader.read_field('MeanV').permute([1, 0])[:, :]
        self.MaxV = dataloader.read_field('MaxV').permute([1, 0])[:, :]
        self.MinV = dataloader.read_field('MinV').permute([1, 0])[:, :]
        self.StdV = dataloader.read_field('StdV').permute([1, 0])[:, :]

        self.inputDim = inputDim
        self.outputDim = outputDim
        print('The shape of the input data is {} and output data is {}'.format(self.x_data.shape, self.y_data.shape))

    def make_loader(self, n_sample, batch_size, start=0, train=True):
        # Sequence = np.delete(np.arange(0, n_sample, 1), np.arange(0, n_sample, 10))
        Sequence = np.arange(0, n_sample, 1)
        # Xs = self.x_data[start:start + n_sample, self.nSlice:, :]
        # ys = self.y_data[start:start + n_sample, self.nSlice:, :]
        Xs = self.x_data[Sequence, self.nSlice:, :]
        ys = self.y_data[Sequence, self.nSlice:, :]
        Xs = Xs[:, ::self.sub_t, :]
        ys = ys[:, ::self.sub_t, :]
        PDE_weights = self.weights_data
        PDE_weights_test = self.weights_data_test

        Xs_test = self.x_data_test[:, self.nSlice:, :]
        ys_test = self.y_data_test[:, self.nSlice:, :]
        Xs_test = Xs_test[:, ::self.sub_t, :]
        ys_test = ys_test[:, ::self.sub_t, :]
        # Xs[:, 1:5, :] = (-1) * Xs[:, 1:5, :]
        print('Shape of input:{};Shape of output:{}'.format(Xs.shape, ys.shape))

        # # Transformation of PGrid
        # ParaGrid = self.PGrid[0:n_sample, :]
        # ParaGrid = ParaGrid.unsqueeze(-1).permute([0, 2, 1]).repeat([1, self.T, 1])
        # MGrid, KGrid, CGrid = ParaGrid[:, :, 0], ParaGrid[:, :, 1], ParaGrid[:, :, 2]
        # print('Shape of each Para Grid is', MGrid.shape)
        #
        # if self.new:
        #     gridt = torch.tensor(np.linspace(0, 2, self.T), dtype=torch.float)
        # else:
        #     gridt = torch.tensor(np.linspace(0, 2, self.T + 1)[1:], dtype=torch.float)
        # gridt = gridt.reshape(1, self.T)
        # print('Shape of gridt:{}'.format(gridt.shape))
        # print('Warning: Real Time length is encoded in the DOF_Loader as a constant, 2 in this case')
        # # The new=False move makes sense if it is aimed to skip the time=0 step
        #
        # Xs = Xs.reshape(n_sample, self.T)
        # print('Shape of input with extension:{}'.format(Xs.shape))
        # Xs = torch.stack([Xs, gridt.repeat([n_sample, 1]), MGrid, KGrid, CGrid], dim=2)
        # print('Shape of input with grid:{}'.format(Xs.shape))
        dataset = torch.utils.data.TensorDataset(Xs, ys, PDE_weights)
        dataset_test = torch.utils.data.TensorDataset(Xs_test, ys_test, PDE_weights_test)
        if train:
            loader_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
        else:
            loader_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
            loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
        ToOneV = torch.stack([self.MeanV, self.MaxV, self.MinV, self.StdV], dim=2).to(torch.device('cuda:0'))
        print('Shape of input ToOneVector:', ToOneV.shape)
        return loader_train, loader_test, ToOneV, self.W2, self.Eigens2, self.TrackDOFs, self.Nloc


class FES_Loader_WithVirtualData(object):
    def __init__(self, datapath, weights_datapath, datapath_test, weights_datapath_test, datapath_virtual, weights_datapath_virtual, Structure_datapath, nt, nSlice,
                 sub_t, new, inputDim, outputDim, ComDevice):
        dataloader = MatReader(datapath)
        dataloader_test = MatReader(datapath_test)
        dataloader_weight = MatReader(weights_datapath)
        dataloader_weight_test = MatReader(weights_datapath_test)
        dataloader_virtual = MatReader(datapath_virtual)
        dataloader_weight_virtual = MatReader(weights_datapath_virtual)
        dataloader_structure = MatReader(Structure_datapath)
        self.sub_t = sub_t
        self.T = nt // sub_t
        self.new = new
        self.nSlice = nSlice
        if new:
            self.T += 1
        self.x_data = dataloader.read_field('input').permute([2, 1, 0])[:, :nt, :]
        self.y_data = dataloader.read_field('output').permute([2, 1, 0])[:, :nt, :]
        self.x_data_test = dataloader_test.read_field('input').permute([2, 1, 0])[:, :nt, :]
        self.y_data_test = dataloader_test.read_field('output').permute([2, 1, 0])[:, :nt, :]
        self.weights_data = dataloader_weight.read_field('Weights').permute([1, 0])
        self.x_data_virtual = dataloader_virtual.read_field('input').permute([2, 1, 0])[:, :nt, :]
        self.weights_data_test = dataloader_weight_test.read_field('Weights').permute([1, 0])
        self.weights_data_virtual = dataloader_weight_virtual.read_field('Weights').permute([1, 0])

        self.W2 = dataloader_structure.read_field('W2').permute([1, 0]).to(ComDevice)
        self.Eigens2 = dataloader_structure.read_field('Eigens2').permute([1, 0]).to(ComDevice)
        self.TrackDOFs = dataloader_structure.read_field('TrackDOFs').permute([1, 0])
        self.Nloc = dataloader_structure.read_field('Nloc').permute([1, 0]).int().to(ComDevice)
        self.ComDevice = ComDevice
        print('Structure Information Warning:W2_{};Eigens2_{};TrackDOFs_{};Nloc_{}'.format(self.W2.shape,
                                                                                           self.Eigens2.shape,
                                                                                           self.TrackDOFs.shape,
                                                                                           self.Nloc.shape))
        # Providing two different data ToOne action options
        self.MeanV = dataloader.read_field('MeanV').permute([1, 0])[:, :]
        self.MaxV = dataloader.read_field('MaxV').permute([1, 0])[:, :]
        self.MinV = dataloader.read_field('MinV').permute([1, 0])[:, :]
        self.StdV = dataloader.read_field('StdV').permute([1, 0])[:, :]

        self.inputDim = inputDim
        self.outputDim = outputDim
        print('The shape of the input data is {} and output data is {}'.format(self.x_data.shape, self.y_data.shape))

    def make_loader(self, n_sample, n_sample_virtual, batch_size, batch_size_virtual, start=0, train=True):
        # Sequence = np.delete(np.arange(0, n_sample, 1), np.arange(0, n_sample, 10))
        Sequence = np.arange(0, n_sample, 1)
        Xs = self.x_data[start:start + n_sample, self.nSlice:, :]
        ys = self.y_data[start:start + n_sample, self.nSlice:, :]
        # Xs = self.x_data[Sequence, self.nSlice:, :]
        # ys = self.y_data[Sequence, self.nSlice:, :]
        Xs = Xs[:, ::self.sub_t, :]
        ys = ys[:, ::self.sub_t, :]
        PDE_weights = self.weights_data[start:start + n_sample, :]
        PDE_weights_test = self.weights_data_test
        PDE_weights_virtual = self.weights_data_virtual

        Xs_test = self.x_data_test[:, self.nSlice:, :]
        ys_test = self.y_data_test[:, self.nSlice:, :]
        Xs_test = Xs_test[:, ::self.sub_t, :]
        ys_test = ys_test[:, ::self.sub_t, :]
        Xs_virtual = self.x_data_virtual[: n_sample_virtual, self.nSlice:, :]
        print('Shape of input:{};Shape of output:{}'.format(Xs.shape, ys.shape))
        print('And {} ansatz of virtual data has been used'.format(Xs_virtual.size(0)))

        dataset = torch.utils.data.TensorDataset(Xs, ys, PDE_weights)
        dataset_test = torch.utils.data.TensorDataset(Xs_test, ys_test, PDE_weights_test)
        dataset_virtual = torch.utils.data.TensorDataset(Xs_virtual)
        if train:
            loader_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
            loader_virtual = torch.utils.data.DataLoader(dataset_virtual, batch_size=batch_size_virtual, shuffle=True)
        else:
            loader_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
            loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
            loader_virtual = torch.utils.data.DataLoader(dataset_virtual, batch_size=batch_size_virtual, shuffle=True)
        ToOneV = torch.stack([self.MeanV, self.MaxV, self.MinV, self.StdV], dim=2).to(self.ComDevice)
        print('Shape of input ToOneVector:', ToOneV.shape)
        return loader_train, loader_test, loader_virtual, PDE_weights_virtual, ToOneV, self.W2, self.Eigens2, self.TrackDOFs, self.Nloc


class VTCD_Loader(object):
    def __init__(self, datapath, weights_datapath, datapath_test, nt, nSlice, sub_t, new, inputDim, outputDim):
        dataloader = MatReader(datapath)
        dataloader_test = MatReader(datapath_test)
        dataloader_weight = MatReader(weights_datapath)
        self.sub_t = sub_t
        self.T = nt // sub_t
        self.new = new
        self.nSlice = nSlice
        if new:
            self.T += 1
        self.x_data = dataloader.read_field('input').permute([2, 1, 0])[:, :nt, :]
        self.y_data = dataloader.read_field('output').permute([2, 1, 0])[:, :nt, :]
        self.x_data_test = dataloader_test.read_field('input').permute([2, 1, 0])[:, :nt, :]
        self.y_data_test = dataloader_test.read_field('output').permute([2, 1, 0])[:, :nt, :]
        self.weights_data = dataloader_weight.read_field('Weights').permute([1, 0])
        # Providing two different data ToOne action options
        self.MeanV = dataloader.read_field('MeanV').permute([1, 0])[:, :]
        self.MaxV = dataloader.read_field('MaxV').permute([1, 0])[:, :]
        self.MinV = dataloader.read_field('MinV').permute([1, 0])[:, :]
        self.StdV = dataloader.read_field('StdV').permute([1, 0])[:, :]

        self.inputDim = inputDim
        self.outputDim = outputDim
        print('The shape of the input data is {} and output data is {}'.format(self.x_data.shape, self.y_data.shape))

    def make_loader(self, n_sample, batch_size, start=0, train=True):
        # Sequence = np.delete(np.arange(0, n_sample, 1), np.arange(0, n_sample, 10))
        Sequence = np.arange(0, n_sample, 1)
        # Xs = self.x_data[start:start + n_sample, self.nSlice:, :]
        # ys = self.y_data[start:start + n_sample, self.nSlice:, :]
        Xs = self.x_data[Sequence, self.nSlice:, :]
        ys = self.y_data[Sequence, self.nSlice:, :]
        Xs = Xs[:, ::self.sub_t, :]
        ys = ys[:, ::self.sub_t, :]
        PDE_weights = self.weights_data
        Xs_test = self.x_data_test[:, self.nSlice:, :]
        ys_test = self.y_data_test[:, self.nSlice:, :]
        Xs_test = Xs_test[:, ::self.sub_t, :]
        ys_test = ys_test[:, ::self.sub_t, :]
        # Xs[:, 1:5, :] = (-1) * Xs[:, 1:5, :]
        print('Shape of input:{};Shape of output:{}'.format(Xs.shape, ys.shape))


        dataset = torch.utils.data.TensorDataset(Xs, ys, PDE_weights)
        dataset_test = torch.utils.data.TensorDataset(Xs_test, ys_test)
        if train:
            loader_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
        else:
            loader_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
            loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
        ToOneV = torch.stack([self.MeanV, self.MaxV, self.MinV, self.StdV], dim=2)
        print('Shape of input ToOneVector:', ToOneV.shape)
        #
        return loader_train, loader_test, ToOneV


class VTCD_Loader_Variant1(object):
    def __init__(self, datapath, datapath_test, nt, nSlice, sub_t, new, inputDim, outputDim):
        dataloader = MatReader(datapath)
        datalaoder_test = MatReader(datapath_test)
        self.sub_t = sub_t
        self.T = nt // sub_t
        self.new = new
        self.nSlice = nSlice
        if new:
            self.T += 1
        self.x_data = dataloader.read_field('input').permute([2, 1, 0])[:, :nt, :]
        self.y_data = dataloader.read_field('output').permute([2, 1, 0])[:, :nt, :]
        self.x_data_test = datalaoder_test.read_field('input').permute([2, 1, 0])[:, :nt, :]
        self.y_data_test = datalaoder_test.read_field('output').permute([2, 1, 0])[:, :nt, :]
        # Providing two different data ToOne action options
        self.MeanV = dataloader.read_field('MeanV').permute([1, 0])[:, :]
        self.MaxV = dataloader.read_field('MaxV').permute([1, 0])[:, :]
        self.MinV = dataloader.read_field('MinV').permute([1, 0])[:, :]
        self.StdV = dataloader.read_field('StdV').permute([1, 0])[:, :]

        self.inputDim = inputDim
        self.outputDim = outputDim
        print('The shape of the input data is {} and output data is {}'.format(self.x_data.shape, self.y_data.shape))

    def make_loader(self, n_sample, batch_size, start=0, train=True):
        # Sequence = np.delete(np.arange(0, n_sample, 1), np.arange(0, n_sample, 10))
        Sequence = np.arange(0, n_sample, 1)
        # Xs = self.x_data[start:start + n_sample, self.nSlice:, :]
        # ys = self.y_data[start:start + n_sample, self.nSlice:, :]
        Xs = self.x_data[Sequence, self.nSlice:, :]
        ys = self.y_data[Sequence, self.nSlice:, :]
        Xs = Xs[:, ::self.sub_t, :]
        ys = ys[:, ::self.sub_t, :]
        Xs_test = self.x_data_test[:, self.nSlice:, :]
        ys_test = self.y_data_test[:, self.nSlice:, :]
        Xs_test = Xs_test[:, ::self.sub_t, :]
        ys_test = ys_test[:, ::self.sub_t, :]
        # Xs[:, 1:5, :] = (-1) * Xs[:, 1:5, :]
        print('Shape of input:{};Shape of output:{}'.format(Xs.shape, ys.shape))

        dataset = torch.utils.data.TensorDataset(Xs, ys)
        dataset_test = torch.utils.data.TensorDataset(Xs_test, ys_test)
        if train:
            loader_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
        else:
            loader_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
            loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
        ToOneV = torch.stack([self.MeanV, self.MaxV, self.MinV, self.StdV], dim=2)
        print('Shape of input ToOneVector:', ToOneV.shape)
        #
        return loader_train, loader_test, ToOneV


class VTCD_Loader_WithVirtualData(object):
    def __init__(self, datapath, weights_datapath, datapath_test, datapath_virtual, weights_datapath_virtual, nt, nSlice, sub_t, new, inputDim, outputDim):
        dataloader = MatReader(datapath)
        dataloader_test = MatReader(datapath_test)
        dataloader_weight = MatReader(weights_datapath)
        dataloader_virtual = MatReader(datapath_virtual)
        dataloader_weight_virtual = MatReader(weights_datapath_virtual)
        self.sub_t = sub_t
        self.T = nt // sub_t
        self.new = new
        self.nSlice = nSlice
        if new:
            self.T += 1
        self.x_data = dataloader.read_field('input').permute([2, 1, 0])[:, :nt, :]
        self.y_data = dataloader.read_field('output').permute([2, 1, 0])[:, :nt, :]
        self.x_data_test = dataloader_test.read_field('input').permute([2, 1, 0])[:, :nt, :]
        self.y_data_test = dataloader_test.read_field('output').permute([2, 1, 0])[:, :nt, :]
        self.weights_data = dataloader_weight.read_field('Weights').permute([1, 0])
        self.x_data_virtual = dataloader_virtual.read_field('input').permute([2, 1, 0])[:, :nt, :]
        self.weights_data_virtual = dataloader_weight_virtual.read_field('Weights').permute([1, 0])
        # Providing two different data ToOne action options
        self.MeanV = dataloader.read_field('MeanV').permute([1, 0])[:, :]
        self.MaxV = dataloader.read_field('MaxV').permute([1, 0])[:, :]
        self.MinV = dataloader.read_field('MinV').permute([1, 0])[:, :]
        self.StdV = dataloader.read_field('StdV').permute([1, 0])[:, :]

        self.inputDim = inputDim
        self.outputDim = outputDim
        print('The shape of the input data is {} and output data is {}'.format(self.x_data.shape, self.y_data.shape))

    def make_loader(self, n_sample, n_sample_virtual, batch_size, batch_size_virtual, start=0, train=True):
        Xs = self.x_data[start:start + n_sample, self.nSlice:, :]
        ys = self.y_data[start:start + n_sample, self.nSlice:, :]
        Xs = Xs[:, ::self.sub_t, :]
        ys = ys[:, ::self.sub_t, :]
        PDE_weights = self.weights_data[start:start + n_sample, :]
        PDE_weights_Virtual = self.weights_data_virtual[start:start + n_sample, :]
        Xs_test = self.x_data_test[:, self.nSlice:, :]
        ys_test = self.y_data_test[:, self.nSlice:, :]
        Xs_test = Xs_test[:, ::self.sub_t, :]
        ys_test = ys_test[:, ::self.sub_t, :]
        Xs_virtual = self.x_data_virtual[: n_sample_virtual, self.nSlice:, :]
        print('Shape of input:{};Shape of output:{}'.format(Xs.shape, ys.shape))
        print('And {} ansatz of virtual data has been used'.format(Xs_virtual.size(0)))

        dataset = torch.utils.data.TensorDataset(Xs, ys, PDE_weights)
        dataset_test = torch.utils.data.TensorDataset(Xs_test, ys_test)
        dataset_virtual = torch.utils.data.TensorDataset(Xs_virtual)
        if train:
            loader_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
            loader_virtual = torch.utils.data.DataLoader(dataset_virtual, batch_size=batch_size_virtual, shuffle=True)
        else:
            loader_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
            loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
            loader_virtual = torch.utils.data.DataLoader(dataset_virtual, batch_size=batch_size_virtual, shuffle=True)
        ToOneV = torch.stack([self.MeanV, self.MaxV, self.MinV, self.StdV], dim=2)
        print('Shape of input ToOneVector:', ToOneV.shape)
        return loader_train, loader_test, loader_virtual, PDE_weights_Virtual, ToOneV


class BSA_Loader_WithVirtualData(object):
    def __init__(self, datapath, weights_datapath, datapath_test, weights_datapath_test, datapath_virtual, weights_datapath_virtual, Structure_datapath, nt, nSlice,
                 sub_t, new, inputDim, outputDim, ComDevice):
        dataloader = MatReader(datapath)
        dataloader_test = MatReader(datapath_test)
        dataloader_weight = MatReader(weights_datapath)
        dataloader_weight_test = MatReader(weights_datapath_test)
        dataloader_virtual = MatReader(datapath_virtual)
        dataloader_weight_virtual = MatReader(weights_datapath_virtual)
        dataloader_structure = MatReader(Structure_datapath)
        self.sub_t = sub_t
        self.T = nt // sub_t
        self.new = new
        self.nSlice = nSlice
        if new:
            self.T += 1
        self.x_data = dataloader.read_field('input').permute([2, 1, 0])[:, :nt, :]
        self.y_data = dataloader.read_field('output').permute([2, 1, 0])[:, :nt, :]
        self.x_data_test = dataloader_test.read_field('input').permute([2, 1, 0])[:, :nt, :]
        self.y_data_test = dataloader_test.read_field('output').permute([2, 1, 0])[:, :nt, :]
        self.weights_data = dataloader_weight.read_field('Weights').permute([1, 0])
        self.x_data_virtual = dataloader_virtual.read_field('input').permute([2, 1, 0])[:, :nt, :]
        self.weights_data_test = dataloader_weight_test.read_field('Weights').permute([1, 0])
        self.weights_data_virtual = dataloader_weight_virtual.read_field('Weights').permute([1, 0])

        self.W2_CX = dataloader_structure.read_field('W2_CX').permute([1, 0]).to(ComDevice)
        self.W2_CY = dataloader_structure.read_field('W2_CY').permute([1, 0]).to(ComDevice)
        self.W2_CZ = dataloader_structure.read_field('W2_CZ').permute([1, 0]).to(ComDevice)
        self.Eigens2 = dataloader_structure.read_field('Eigens2').permute([1, 0]).to(ComDevice)
        self.TrackDOFs = dataloader_structure.read_field('TrackDOFs').permute([1, 0])
        self.Nloc = dataloader_structure.read_field('Nloc').permute([1, 0]).int().to(ComDevice)
        self.ComDevice = ComDevice
        print('Structure Information Warning:W2_{};Eigens2_{};TrackDOFs_{};Nloc_{}'.format(self.W2_CX.shape,
                                                                                           self.Eigens2.shape,
                                                                                           self.TrackDOFs.shape,
                                                                                           self.Nloc.shape))
        # Providing two different data ToOne action options
        self.MeanV = dataloader.read_field('MeanV').permute([1, 0])[:, :]
        self.MaxV = dataloader.read_field('MaxV').permute([1, 0])[:, :]
        self.MinV = dataloader.read_field('MinV').permute([1, 0])[:, :]
        self.StdV = dataloader.read_field('StdV').permute([1, 0])[:, :]

        self.inputDim = inputDim
        self.outputDim = outputDim
        print('The shape of the input data is {} and output data is {}'.format(self.x_data.shape, self.y_data.shape))

    def make_loader(self, n_sample, n_sample_virtual, batch_size, batch_size_virtual, start=0, train=True):
        # Sequence = np.delete(np.arange(0, n_sample, 1), np.arange(0, n_sample, 10))
        Sequence = np.arange(0, n_sample, 1)
        Xs = self.x_data[start:start + n_sample, self.nSlice:, :]
        ys = self.y_data[start:start + n_sample, self.nSlice:, :]
        # Xs = self.x_data[Sequence, self.nSlice:, :]
        # ys = self.y_data[Sequence, self.nSlice:, :]
        Xs = Xs[:, ::self.sub_t, :]
        ys = ys[:, ::self.sub_t, :]
        PDE_weights = self.weights_data[start:start + n_sample, :]
        PDE_weights_test = self.weights_data_test
        PDE_weights_virtual = self.weights_data_virtual

        Xs_test = self.x_data_test[:, self.nSlice:, :]
        ys_test = self.y_data_test[:, self.nSlice:, :]
        Xs_test = Xs_test[:, ::self.sub_t, :]
        ys_test = ys_test[:, ::self.sub_t, :]
        # Xs_virtual = self.x_data_virtual[: n_sample_virtual, :, :]
        Xs_virtual = self.x_data_virtual[: n_sample_virtual, self.nSlice:, :]
        print('Shape of input:{};Shape of output:{}'.format(Xs.shape, ys.shape))
        print('And {} ansatz of virtual data has been used'.format(Xs_virtual.size(0)))

        dataset = torch.utils.data.TensorDataset(Xs, ys, PDE_weights)
        dataset_test = torch.utils.data.TensorDataset(Xs_test, ys_test, PDE_weights_test)
        dataset_virtual = torch.utils.data.TensorDataset(Xs_virtual)
        if train:
            loader_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
            loader_virtual = torch.utils.data.DataLoader(dataset_virtual, batch_size=batch_size_virtual, shuffle=True)
        else:
            loader_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
            loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
            loader_virtual = torch.utils.data.DataLoader(dataset_virtual, batch_size=batch_size_virtual, shuffle=True)
        ToOneV = torch.stack([self.MeanV, self.MaxV, self.MinV, self.StdV], dim=2).to(self.ComDevice)
        print('Shape of input ToOneVector:', ToOneV.shape)
        return loader_train, loader_test, loader_virtual, PDE_weights_virtual, ToOneV, \
               self.W2_CX, self.W2_CY, self.W2_CZ, self.Eigens2, self.TrackDOFs, self.Nloc


class HM_Loader_WithVirtualData(object):
    def __init__(self, datapath, weights_datapath, datapath_test, weights_datapath_test, datapath_virtual, weights_datapath_virtual, Structure_datapath, nt, nSlice,
                 sub_t, new, inputDim, outputDim, ComDevice):
        dataloader = MatReader(datapath)
        dataloader_test = MatReader(datapath_test)
        dataloader_weight = MatReader(weights_datapath)
        dataloader_weight_test = MatReader(weights_datapath_test)
        dataloader_virtual = MatReader(datapath_virtual)
        dataloader_weight_virtual = MatReader(weights_datapath_virtual)
        dataloader_structure = MatReader(Structure_datapath)
        self.sub_t = sub_t
        self.T = nt // sub_t
        self.new = new
        self.nSlice = nSlice
        if new:
            self.T += 1
        self.x_data = dataloader.read_field('input').permute([2, 1, 0])[:, :nt, :]
        self.y_data = dataloader.read_field('output').permute([2, 1, 0])[:, :nt, :]
        self.x_data_test = dataloader_test.read_field('input').permute([2, 1, 0])[:, :nt, :]
        self.y_data_test = dataloader_test.read_field('output').permute([2, 1, 0])[:, :nt, :]
        self.weights_data = dataloader_weight.read_field('Weights').permute([1, 0])
        self.x_data_virtual = dataloader_virtual.read_field('input').permute([2, 1, 0])[:, :nt, :]
        self.weights_data_test = dataloader_weight_test.read_field('Weights').permute([1, 0])
        self.weights_data_virtual = dataloader_weight_virtual.read_field('Weights').permute([1, 0])

        self.W2_1 = dataloader_structure.read_field('W2_1').permute([1, 0]).to(ComDevice)
        self.W2_2 = dataloader_structure.read_field('W2_2').permute([1, 0]).to(ComDevice)
        self.W2_3 = dataloader_structure.read_field('W2_3').permute([1, 0]).to(ComDevice)
        self.Eigens2_1 = dataloader_structure.read_field('Eigens2_1').permute([1, 0]).to(ComDevice)
        self.Eigens2_2 = dataloader_structure.read_field('Eigens2_2').permute([1, 0]).to(ComDevice)
        # self.TrackDOFs = dataloader_structure.read_field('TrackDOFs').permute([1, 0])
        # self.Nloc = dataloader_structure.read_field('Nloc').permute([1, 0]).int().to(ComDevice)
        self.ComDevice = ComDevice
        print('Structure Information Warning:W2_1:{};W2_2:{};W2_3:{};Eigens2_1:{};Eigens2_2:{};'.format(self.W2_1.shape,
                                                                                                        self.W2_2.shape,
                                                                                                        self.W2_3.shape,
                                                                                                        self.Eigens2_1.shape,
                                                                                                        self.Eigens2_2.shape))
        # Providing two different data ToOne action options
        self.MeanV = dataloader.read_field('MeanV').permute([1, 0])[:, :]
        self.MaxV = dataloader.read_field('MaxV').permute([1, 0])[:, :]
        self.MinV = dataloader.read_field('MinV').permute([1, 0])[:, :]
        self.StdV = dataloader.read_field('StdV').permute([1, 0])[:, :]

        self.inputDim = inputDim
        self.outputDim = outputDim
        print('The shape of the input data is {} and output data is {}'.format(self.x_data.shape, self.y_data.shape))

    def make_loader(self, n_sample, n_sample_virtual, batch_size, batch_size_virtual, start=0, train=True):
        # Sequence = np.delete(np.arange(0, n_sample, 1), np.arange(0, n_sample, 10))
        Sequence = np.arange(0, n_sample, 1)
        Xs = self.x_data[start:start + n_sample, self.nSlice:, :]
        ys = self.y_data[start:start + n_sample, self.nSlice:, :]

        # Xs = self.x_data[Sequence, self.nSlice:, :]
        # ys = self.y_data[Sequence, self.nSlice:, :]
        Xs = Xs[:, ::self.sub_t, :]
        ys = ys[:, ::self.sub_t, :]
        PDE_weights = self.weights_data[start:start + n_sample, :]
        PDE_weights_test = self.weights_data_test
        PDE_weights_virtual = self.weights_data_virtual

        Xs_test = self.x_data_test[:, self.nSlice:, :]
        ys_test = self.y_data_test[:, self.nSlice:, :]
        Xs_test = Xs_test[:, ::self.sub_t, :]
        ys_test = ys_test[:, ::self.sub_t, :]
        Xs_virtual = self.x_data_virtual[: n_sample_virtual, self.nSlice:, :]
        print('Shape of input:{};Shape of output:{}'.format(Xs.shape, ys.shape))
        print('And {} ansatz of virtual data has been used'.format(Xs_virtual.size(0)))

        dataset = torch.utils.data.TensorDataset(Xs, ys, PDE_weights)
        dataset_test = torch.utils.data.TensorDataset(Xs_test, ys_test, PDE_weights_test)
        dataset_virtual = torch.utils.data.TensorDataset(Xs_virtual)
        if train:
            loader_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
            loader_virtual = torch.utils.data.DataLoader(dataset_virtual, batch_size=batch_size_virtual, shuffle=True)
        else:
            loader_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
            loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
            loader_virtual = torch.utils.data.DataLoader(dataset_virtual, batch_size=batch_size_virtual, shuffle=True)
        ToOneV = torch.stack([self.MeanV, self.MaxV, self.MinV, self.StdV], dim=2).to(self.ComDevice)
        print('Shape of input ToOneVector:', ToOneV.shape)
        return loader_train, loader_test, loader_virtual, PDE_weights_virtual, ToOneV, \
               self.W2_1, self.W2_2, self.W2_3, self.Eigens2_1, self.Eigens2_2