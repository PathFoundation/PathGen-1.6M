import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# vpt
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from functools import reduce
from operator import mul


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()

        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x)  # N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(self, conf, dropout_v=0.0, nonlinear=True, passing_v=False,
                 confounder_path=False):  # K, L, N
        super(BClassifier, self).__init__()
        input_size=conf.D_feat
        output_class=conf.n_class
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, conf.D_inner), nn.ReLU(), nn.Linear(conf.D_inner, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, conf.D_inner)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()

        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)


    def forward(self, feats, c):  # N x K, N x C
        device = feats.device
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0,
                                  descending=True)  # sort class scores along the instance dimension, m_indices in shape N x C
        # print(m_indices.shape)
        m_feats = torch.index_select(feats, dim=0,
                                     index=m_indices[0, :])  # select critical instances, m_feats in shape C x K
        q_max = self.q(m_feats)  # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0,
                                        1))  # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device))  # normalize attention scores, A in shape N x C,
        A = A.transpose(0, 1)

        A_out = A
        A = F.softmax(A, dim=-1)
        B = torch.mm(A, V)  # compute bag representation, B in shape C x V
        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V

        C = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)
        return C, A_out, B


class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x):
        feats, classes = self.i_classifier(x[0])
        # print(feats)
        prediction_bag, A, B = self.b_classifier(feats, classes)
        return classes, prediction_bag, A
