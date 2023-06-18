import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self,size=256,conv='2d'):
        super(ConvBlock, self).__init__()
        if conv == '2d':
            self.conv1 = nn.Conv2d(21, size, 3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(size)
        elif conv == '1d':
            self.conv1 = nn.Conv1d(size, size, 3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm1d(size)

    def forward(self, s):
        if len(s.shape) == 3:
            s = s.view(-1, 21, 8, 8)
        s = s.float()
        s = F.relu(self.bn1(self.conv1(s)))
        return s


class ResBlock(nn.Module):
    def __init__(self, inplanes=256, planes=256, stride=1, downsample=None,conv='2d'):
        super(ResBlock, self).__init__()
        if conv == '2d':
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
        elif conv == '1d':
            self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
            self.bn1 = nn.BatchNorm1d(planes)
            self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
            self.bn2 = nn.BatchNorm1d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(256, 1, kernel_size=1)  # value head
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8 * 8, 64)
        self.fc2 = nn.Linear(64, 1)

        self.conv1 = nn.Conv2d(256, 128, kernel_size=1)  # policy head
        self.bn1 = nn.BatchNorm2d(128)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(8 * 8 * 128, 8 * 8 * 73)

    def forward(self, s):
        v = F.relu(self.bn(self.conv(s)))  # value head
        v = v.view(-1, 8 * 8)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))

        p = F.relu(self.bn1(self.conv1(s)))  # policy head
        p = p.view(-1, 8 * 8 * 128)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v


class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv = ConvBlock()
        for block in range(19):
            setattr(self, "res_%i" % block, ResBlock())
        self.outblock = OutBlock()

    def forward(self, s):
        s = self.conv(s)
        for block in range(19):
            s = getattr(self, "res_%i" % block)(s)
        p, v = self.outblock(s)
        return p, v


class AlphaLoss(nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy *
                                   (1e-6 + y_policy.float()).float().log()), 1)
        value_error = value_error.view(-1).float().mean()
        policy_error = policy_error.view(-1).float().mean()
        return value_error, policy_error

def move_acc(prediction, target):
    move_accs, best_acc = [], []
    # Assume prediction and target are of size (batch_size, 4672)
    for pred, real in zip(prediction, target):
        pred_argmax = torch.argmax(pred).item()
        real_argmax = torch.argmax(real).item()
        best_acc.append(1 if pred_argmax == real_argmax else 0)
        # find the indices of the nonzero elements
        real_nonzero_indices = real.nonzero(as_tuple=True)[0].tolist()
        move_accs.append(1 if pred_argmax in real_nonzero_indices else 0)
        # print('pred: ',pred_argmax)
        # print('real: ',real_argmax)
        # print('real_nonzero_indices: ',real_nonzero_indices)
    result = np.sum(move_accs) / len(move_accs)
    best_acc_result = np.sum(best_acc) / len(best_acc)
    return result, best_acc_result