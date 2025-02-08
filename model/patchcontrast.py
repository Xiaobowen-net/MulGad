import torch
import numpy as np
from torch import nn
import math
import torch.nn.functional as F

class patchcontrast(torch.nn.Module):
    def __init__(self, device, win_size,temperature,patch_len,batch_size):
        super(patchcontrast, self).__init__()
        self.patch_len = patch_len
        self.stride = patch_len
        self.win_size =win_size
        self.patch_num = int((self.win_size - self.patch_len)/self.stride + 1)+1
        self.temperature = temperature
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.lamda = nn.Parameter(torch.ones(1,self.patch_num).repeat(batch_size,self.patch_num, 1))
        self.lsoftmax = torch.nn.LogSoftmax(dim=-1)
        self.device = device
        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

        self.W = nn.Linear(self.patch_num, self.patch_num)

    def forward(self, z):
        batch_size = z.shape[0]
        z = z.permute(0, 2, 1)
        z = self.padding_patch_layer(z)
        representations = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        representations = representations.permute(0, 2, 3 , 1).reshape(batch_size,self.patch_num,-1)

        representations_normalized = F.normalize(representations, dim=-1)


        similarity_matrix = self._cosine_similarity(representations_normalized.unsqueeze(2),
                                                     representations_normalized.unsqueeze(1))
        left_matrix = torch.zeros((self.patch_num, self.patch_num)).to(self.device)
        left_matrix[:-1, 1:] = torch.eye(self.patch_num - 1).to(self.device)
        right_matrix = torch.zeros((self.patch_num, self.patch_num)).to(self.device)
        right_matrix[1:, :-1] = torch.eye(self.patch_num - 1).to(self.device)
        matrix = right_matrix+left_matrix
        adjacent_matrix = matrix.repeat(batch_size,1,1)
        pos = torch.sum(torch.exp((similarity_matrix * adjacent_matrix) /self.temperature), dim=-1)


        neg_matrix = torch.ones((batch_size,self.patch_num,self.patch_num)).to(self.device)-(torch.eye(self.patch_num, dtype=torch.int).repeat(batch_size,1,1).to(self.device)+ adjacent_matrix)
        negatives = torch.exp((similarity_matrix * neg_matrix)/self.temperature)

        negatives =torch.sum(negatives * F.softmax(self.lamda,dim=-1), dim=-1)
        loss = torch.sum(torch.log(pos/(pos+negatives)))
        return loss /(-1.0 * self.patch_num * batch_size)