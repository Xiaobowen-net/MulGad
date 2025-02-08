
import torch.nn as nn
class ConvLayer(nn.Module):
    def __init__(self, in_dim):
        super(ConvLayer, self).__init__()
        self.main = nn.Sequential(

            nn.Conv1d(in_dim, 64, 7, 1, 3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(64, 128 , 7, 1, 3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, input):
        output = self.main(input)

        return output
