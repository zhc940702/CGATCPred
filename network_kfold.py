import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from Extra_label_matrix import *
dropout = 0.5
batch_size = 20
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=(1, 3), stride=(1, stride), padding=(0,1), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(1, 3), stride=1, padding=(0,1), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=(1, stride), bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

def _get_hout_res18(hin):
    hout = hin
    hout = math.floor(float(hout-1) / 2 + 1)
    hout = math.floor(float(hout-1) / 2 + 1)
    hout = math.floor(float(hout-1) / 2 + 1)
    hout = math.floor(float(hout-1) / 2 + 1)
    return int(hout)

def _get_wout_res18(win):
    wout = win
    wout = math.floor(float(wout-1) / 2 + 1)
    wout = math.floor(float(wout-1) / 2 + 1)
    wout = math.floor(float(wout-1) / 2 + 1)
    wout = math.floor(float(wout-1) / 2 + 1)
    return int(wout)

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        input = input.float()
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class CNN_GCN(nn.Module):
    pool_kernel = 4

    def __init__(self, drug_label, nfeat, block, num_blocks, hout=1, wout=1):
        super(CNN_GCN, self).__init__()
        self.in_planes = 64
        self.gc1 = GraphConvolution(nfeat, 512)
        self.gc2 = GraphConvolution(512, 1010)
        self.dropout = dropout
        self.relu = nn.LeakyReLU(0.2)
        _adj = extra(drug_label, t=0.5)
        self.A = Parameter(torch.from_numpy(_adj).float())

        hout_r = int(math.floor((hout - self.pool_kernel) / self.pool_kernel) + 1)
        wout_r = int(math.floor((wout - self.pool_kernel) / self.pool_kernel) + 1)
        hout_r = 1 if hout_r <= 0 else hout_r
        wout_r = 1 if wout_r <= 0 else wout_r

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1,3), stride=1, padding=(0,1), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(int(512 * hout_r * wout_r * 7), 1010)
        self.linear2 = nn.Linear(int(1010 + 14), 256)
        self.linear3 = nn.Linear(256, 14)
        self.sigmoid = nn.Sigmoid()
        self.dropout_layer1 = nn.Dropout(self.dropout)
        self.dropout_layer2 = nn.Dropout(self.dropout)
        self.dropout_layer3 = nn.Dropout(self.dropout)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, y):
        x = x.float()
        x = torch.reshape(x, (-1, 1, 7, x.size(1)//7))
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, (1, self.pool_kernel))
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear(x))
        x = self.dropout_layer1(x)

        adj = gen_adj(self.A).detach()
        y = F.relu(self.gc1(y, adj))
        y = self.dropout_layer2(y)
        y = self.gc2(y, adj)
        y = y.transpose(0, 1)
        x1 = torch.matmul(x, y)
        x_y = torch.cat((x, x1), 1)
        x_y = F.relu(self.linear2(x_y))
        x_y = self.dropout_layer3(x_y)
        out = self.linear3(x_y)
        return out

def ResNet18_GCN(drug_label, hin=1, win=3883):
    hout, wout = _get_hout_res18(hin), _get_wout_res18(win)
    nfeat = 300
    return CNN_GCN(drug_label, nfeat, BasicBlock, [2, 2, 2, 2], hout, wout)