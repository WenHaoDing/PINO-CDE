import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from functools import reduce
from functools import partial

from .basics import SpectralConv1d


class FNN1d(nn.Module):
    def __init__(self, modes, width, layers=None):
        super(FNN1d, self).__init__()

        """
        The overall network. It contains several layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        if layers is None:
            layers = [width] * 4

        self.fc0 = nn.Linear(2, layers[0])  # input channel is 2: (a(x), x)

        self.sp_convs = nn.ModuleList([SpectralConv1d(
            in_size, out_size, self.modes1) for in_size, out_size in zip(layers, layers[1:])])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(layers, layers[1:])])

        self.fc1 = nn.Linear(layers[-1], 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        length = len(self.ws)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x)
            x = x1 + x2
            if i != length - 1:
                x = F.relu(x)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class FNN1d_DOF(nn.Module):
    def __init__(self, modes, width, layers=None):
        super(FNN1d_DOF, self).__init__()

        """
        The overall network. It contains several layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        if layers is None:
            layers = [width] * 2

        self.fc0 = nn.Linear(5, layers[0])  # input channel is 2: (a(x), x)

        self.sp_convs = nn.ModuleList([SpectralConv1d(
            in_size, out_size, self.modes1) for in_size, out_size in zip(layers, layers[1:])])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(layers, layers[1:])])

        self.fc1 = nn.Linear(layers[-1], 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        length = len(self.ws)
        print(x.shape)
        os.system('pause')

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x)
            x = x1 + x2
            if i != length - 1:
                x = F.elu(x)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        return x


class FNN1d_VTCD(nn.Module):
    def __init__(self, modes, width, fc_dim, inputDim, outputDim, layers=None):
        super(FNN1d_VTCD, self).__init__()

        """
        The overall network. It contains several layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2.

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.inputDim = inputDim
        self.outputDim = outputDim

        if layers is None:
            layers = [width] * 4

        self.fc0 = nn.Linear(self.inputDim, layers[0])  # input channel is 2: (a(x), x)

        self.sp_convs = nn.ModuleList([SpectralConv1d(
            in_size, out_size, self.modes1) for in_size, out_size in zip(layers, layers[1:])])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(layers, layers[1:])])

        self.fc_width = fc_dim
        self.fc1 = nn.Linear(layers[-1], self.fc_width)
        self.fc2 = nn.Linear(self.fc_width, self.fc_width)
        self.fc3 = nn.Linear(self.fc_width, self.outputDim)
        self.bn1 = nn.BatchNorm1d(2500)
        self.bnW = nn.BatchNorm1d(width)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        length = len(self.ws)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x)
            x = x1 + x2
            if i != length - 1:
                x = F.elu(x)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.fc3(x)
        return x


class FNN1d_VTCD_GradNorm(nn.Module):
    def __init__(self, modes, width, fc_dim, inputDim, outputDim, task_number, layers=None):
        super(FNN1d_VTCD_GradNorm, self).__init__()

        """
        The overall network. It contains several layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2.

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.inputDim = inputDim
        self.outputDim = outputDim

        if layers is None:
            layers = [width] * 4

        self.fc0 = nn.Linear(self.inputDim, layers[0])  # input channel is 2: (a(x), x)

        self.sp_convs = nn.ModuleList([SpectralConv1d(
            in_size, out_size, self.modes1) for in_size, out_size in zip(layers, layers[1:])])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(layers, layers[1:])])

        self.fc_width = fc_dim
        self.fc1 = nn.Linear(layers[-1], self.fc_width)
        self.fc2 = nn.Linear(self.fc_width, self.fc_width)
        self.fc3 = nn.Linear(self.fc_width, self.outputDim)
        self.bn1 = nn.BatchNorm1d(2500)
        self.bnW = nn.BatchNorm1d(width)
        self.dropout = nn.Dropout(p=0.5)
        self.task_weights = torch.nn.Parameter(torch.ones(task_number).float())

    def forward(self, x):
        length = len(self.ws)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x)
            x = x1 + x2
            if i != length - 1:
                x = F.elu(x)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.fc3(x)
        return x

    def get_last_layer(self):
        return self.fc3

class FNN1d_FES(nn.Module):
    def __init__(self, modes, width, fc_dim, inputDim, outputDim, layers=None):
        super(FNN1d_FES, self).__init__()

        """
        The overall network. It contains several layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.inputDim = inputDim
        self.outputDim = outputDim

        if layers is None:
            layers = [width] * 3

        self.fc0 = nn.Linear(self.inputDim, layers[0])  # input channel is 2: (a(x), x)

        self.sp_convs = nn.ModuleList([SpectralConv1d(
            in_size, out_size, self.modes1) for in_size, out_size in zip(layers, layers[1:])])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(layers, layers[1:])])

        self.fc_width = fc_dim
        self.fc1 = nn.Linear(layers[-1], self.fc_width)
        self.fc2 = nn.Linear(self.fc_width, self.fc_width)
        self.fc3 = nn.Linear(self.fc_width, self.outputDim)

        self.bn1 = nn.BatchNorm1d(2500)
        self.bnW = nn.BatchNorm1d(width)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        length = len(self.ws)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x)
            x = x1 + x2
            if i != length - 1:
                x = F.elu(x)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.fc3(x)
        return x


class FNN1d_BSA(nn.Module):
    def __init__(self, modes, width, fc_dim, inputDim, outputDim, layers=None):
        super(FNN1d_BSA, self).__init__()

        """
        The overall network. It contains several layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.inputDim = inputDim
        self.outputDim = outputDim

        if layers is None:
            layers = [width] * 3

        self.fc0 = nn.Linear(self.inputDim, layers[0])  # input channel is 2: (a(x), x)

        self.sp_convs = nn.ModuleList([SpectralConv1d(
            in_size, out_size, self.modes1) for in_size, out_size in zip(layers, layers[1:])])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(layers, layers[1:])])

        self.fc_width = fc_dim
        self.fc1 = nn.Linear(layers[-1], self.fc_width)
        self.fc2 = nn.Linear(self.fc_width, self.fc_width)
        self.fc3 = nn.Linear(self.fc_width, self.outputDim)

        self.bn1 = nn.BatchNorm1d(2500)
        self.bnW = nn.BatchNorm1d(width)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        length = len(self.ws)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x)
            x = x1 + x2
            if i != length - 1:
                x = F.elu(x)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.fc3(x)
        return x


class FNN1d_BSA_GradNorm(nn.Module):
    def __init__(self, modes, width, fc_dim, inputDim, outputDim, task_number, layers=None):
        super(FNN1d_BSA_GradNorm, self).__init__()

        """
        The overall network. It contains several layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.inputDim = inputDim
        self.outputDim = outputDim

        if layers is None:
            layers = [width] * 3

        self.fc0 = nn.Linear(self.inputDim, layers[0])  # input channel is 2: (a(x), x)

        self.sp_convs = nn.ModuleList([SpectralConv1d(
            in_size, out_size, self.modes1) for in_size, out_size in zip(layers, layers[1:])])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(layers, layers[1:])])

        self.fc_width = fc_dim
        self.fc1 = nn.Linear(layers[-1], self.fc_width)
        self.fc2 = nn.Linear(self.fc_width, self.fc_width)
        self.fc3 = nn.Linear(self.fc_width, self.outputDim)

        self.bn1 = nn.BatchNorm1d(2500)
        self.bnW = nn.BatchNorm1d(width)
        self.dropout = nn.Dropout(p=0.5)
        self.task_weights = torch.nn.Parameter(torch.ones(task_number).float())

    def forward(self, x):
        length = len(self.ws)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x)
            x = x1 + x2
            if i != length - 1:
                x = F.elu(x)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.fc3(x)
        return x

    def get_last_layer(self):
        return self.fc3

class FNN1d_ANN(nn.Module):
    def __init__(self, inputDim, outputDim, layers=None):
        super(FNN1d_ANN, self).__init__()

        """
        Simple fully connected networks. It contains several layers of ordinary neural layers.
        
        """

        self.inputDim = inputDim
        self.outputDim = outputDim

        self.fc_width = 48
        self.fc0 = nn.Linear(self.inputDim, self.fc_width)
        self.fc1 = nn.Linear(self.fc_width, self.fc_width)
        self.fc2 = nn.Linear(self.fc_width, self.fc_width)
        self.fc3 = nn.Linear(self.fc_width, self.fc_width)
        self.fc4 = nn.Linear(self.fc_width, self.outputDim)
        # self.bn1 = nn.BatchNorm1d(2500)
        # self.bnW = nn.BatchNorm1d(width)

    def forward(self, x):
        x = self.fc0(x)
        x = F.elu(x)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.fc3(x)
        x = F.elu(x)
        x = self.fc4(x)
        return x


class FNN1d_HM(nn.Module):
    def __init__(self, modes, width, fc_dim, inputDim, outputDim, task_number, layers=None):
        super(FNN1d_HM, self).__init__()

        """
        The overall network. It contains several layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.inputDim = inputDim
        self.outputDim = outputDim

        if layers is None:
            layers = [width] * 4

        self.fc0 = nn.Linear(self.inputDim, layers[0])  # input channel is 2: (a(x), x)

        self.sp_convs = nn.ModuleList([SpectralConv1d(
            in_size, out_size, self.modes1) for in_size, out_size in zip(layers, layers[1:])])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(layers, layers[1:])])

        self.fc_width = fc_dim
        self.fc1 = nn.Linear(layers[-1], self.fc_width)
        self.fc2 = nn.Linear(self.fc_width, self.fc_width)
        self.fc3 = nn.Linear(self.fc_width, self.fc_width)
        self.fc4 = nn.Linear(self.fc_width, self.outputDim)

        self.bn1 = nn.BatchNorm1d(2500)
        self.bnW = nn.BatchNorm1d(width)
        self.dropout = nn.Dropout(p=0.5)
        self.task_weights = torch.nn.Parameter(torch.ones(task_number).float())

    def forward(self, x):
        length = len(self.ws)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x)
            x = x1 + x2
            if i != length - 1:
                x = F.elu(x)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.fc3(x)
        x = F.elu(x)
        x = self.fc4(x)
        return x

    def get_last_layer(self):
        return self.fc4


class CNN_GRU(nn.Module):
    def __init__(self, channel, input_dim, output_dim, kernel_size):
        super(CNN_GRU, self).__init__()
        self.channel = channel
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.padding = int(kernel_size / 2 - 1)
        self.CNN1 = nn.Conv1d(in_channels=input_dim, out_channels=20, kernel_size=self.kernel_size,
                              stride=1,
                              padding=50)
        self.CNN2 = nn.Conv1d(in_channels=20, out_channels=40, kernel_size=self.kernel_size,
                              stride=1,
                              padding=49)
        self.CNN3 = nn.Conv1d(in_channels=40, out_channels=50, kernel_size=self.kernel_size,
                              stride=1,
                              padding=50)
        self.CNN4 = nn.Conv1d(in_channels=50, out_channels=50, kernel_size=self.kernel_size,
                              stride=1,
                              padding=49)
        # self.GRU1 = nn.GRU(input_size=50, hidden_size=50, num_layers=5, batch_first=True, bidirectional=False)
        # self.GRU2 = nn.GRU(input_size=50, hidden_size=43, num_layers=5, batch_first=True, bidirectional=False)
        self.GRU1 = nn.GRU(input_size=self.input_dim, hidden_size=30, num_layers=2, batch_first=True,
                           bidirectional=False)
        self.GRU2 = nn.GRU(input_size=30, hidden_size=50, num_layers=2, batch_first=True,
                           bidirectional=False)
        self.GRU3 = nn.GRU(input_size=50, hidden_size=self.output_dim, num_layers=2, batch_first=True,
                           bidirectional=False)
        # self.CNN1 = self.init_CNN(self.CNN1)
        # self.CNN2 = self.init_CNN(self.CNN2)
        # self.CNN3 = self.init_CNN(self.CNN3)
        # self.CNN4 = self.init_CNN(self.CNN4)
        self.GRU1 = self.init_GRU(self.GRU1)
        self.GRU2 = self.init_GRU(self.GRU2)
        self.GRU3 = self.init_GRU(self.GRU3)

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        # x = self.CNN1(x)
        # x = self.CNN2(x)
        # x = self.CNN3(x)
        # x = self.CNN4(x)
        # x = x.permute(0, 2, 1)
        x, _ = self.GRU1(x)
        x, _ = self.GRU2(x)
        x, _ = self.GRU3(x)
        x = x.permute(0, 2, 1)
        return x

    def init_GRU(self, net):
        for name, p in net.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.constant_(p, 0)
        return net

    def init_CNN(self, net):
        nn.init.kaiming_uniform_(net.weight)
        return net
