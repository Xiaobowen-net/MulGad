import torch
import torch.nn as nn
import torch.nn.functional as F
class TC(nn.Module):
    def __init__(self,win_size,d_model,temperature):
        super(TC, self).__init__()
        self.win_size = win_size
        self.lsoftmax = nn.LogSoftmax()
        self.auto_reg = nn.Sequential(nn.Linear(d_model, 128, bias=True))
        self.d_model =d_model
        self.temperature = temperature
    def forward(self, hidden, views):
        batch = views.shape[0]
        re_view = self.auto_reg(hidden)
        dist_pos = F.pairwise_distance(re_view.reshape(batch, -1).unsqueeze(1), views.reshape(batch, -1).unsqueeze(0))/ self.temperature
        nce = torch.sum(torch.diag(self.lsoftmax(-dist_pos)))
        nce /= -1. * batch
        return nce


