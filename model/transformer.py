import torch.nn as nn
from data_factory.augmentations import DataTransform, NegDataTransform
from .TC import TC
from .attn import AttentionLayer
from .conv import ConvLayer
import torch
from .encoder import Attention, EncoderLayer
from .inner_att import innerConv, softmax_kernel_transformation
from .patchcontrast import patchcontrast

class Transformer(nn.Module):
    def __init__(self, input_c,d_ff,patch_len,tau,temperature,win_size,nb_random_features,batch_size,d_model,e_layers,n_heads,dropout,device,activation='gelu'):
        super(Transformer, self).__init__()
        self.dim = input_c
        self.conv = ConvLayer(self.dim)
        self.patch_len = patch_len
        self.tau = tau
        self.temperature = temperature
        self.device = device
        self.win_size = win_size
        self.patch_loss = patchcontrast(self.device,self.win_size,self.tau ,self.patch_len,batch_size)
        self.nb_random_features = nb_random_features
        self.attention = Attention(
            [
                EncoderLayer(
                    AttentionLayer(
                        NodeFormerConv(n_heads,win_size,kernel_transformation=softmax_kernel_transformation, projection_matrix_type='a',nb_random_features=self.nb_random_features),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.TC = TC(self.win_size, d_model, self.temperature)
        self.projection = nn.Linear(128, d_model, bias=True)
        self.catlinear = nn.Linear(d_model * 2 , d_model,bias=True)
        self.rec_linner = nn.Linear(d_model , self.dim)

    def forward(self, x,mode ='test'):
        if mode == 'test':
            enc_out = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
            att_out = self.attention(self.projection(enc_out))
            att_out = self.catlinear(torch.cat((att_out, att_out), dim=2))
            rec = self.rec_linner(att_out)
            return rec, 1

        else:
            enc_out = self.conv(x.permute(0,2,1)).permute(0, 2, 1)
            patch = self.patch_loss(enc_out)
            w_view, s_view = DataTransform(enc_out, self.device)
            w_hidden = self.attention(self.projection(w_view))
            s_hidden = self.attention(self.projection(s_view))

            crossrec = self.TC(s_hidden, w_view) + self.TC(w_hidden, s_view)

            att_out = self.catlinear(torch.cat((s_hidden, w_hidden), dim=2))
            rec = self.rec_linner(att_out)
            return rec, crossrec, patch





