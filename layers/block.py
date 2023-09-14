import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer, DSAttention, FullAttention
from layers.FourierCorrelation import FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class GRN_1d(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        # L2正则：主要用来防止模型过拟合，直观上理解就是L2正则化是对于大数值的权重向量进行严厉惩罚。鼓励参数是较小值
        Gx = torch.norm(x, p=2, dim=(1), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


# block1d的输入和输出都是[batch_size,num_channel,seq_len]这种类似于图像的写法
class Block_1d(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN_1d(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # (N, C, S) -> (N, S, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1)  # (N, S, C) -> (N, C, S)

        x = input + self.drop_path(x)
        return x



class PermuteLayer(nn.Module):
    def __init__(self, dim_order):
        super(PermuteLayer, self).__init__()
        self.dim_order = dim_order

    def forward(self, x):
        return x.permute(self.dim_order)





class lstm_n(nn.Module):

    def __init__(self, configs=None):
        super().__init__()
        self.input_dim = configs.d_model
        self.hidden_dim = configs.d_model
        self.layer_dim = configs.e_layers

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]


class WeightedSum(nn.Module):
    def __init__(self, seq_len, d_model):
        super(WeightedSum, self).__init__()
        self.weights = nn.Parameter(torch.rand(1, seq_len, d_model))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, tensor1, tensor2):
        normalized_weights = torch.softmax(self.weights, dim=1)

        weighted_sum = normalized_weights * tensor1 + (1 - normalized_weights) * tensor2
        return weighted_sum

class fusion_layer(nn.Module):
    def __init__(self, configs, cat_mode='none', attn='prob'):
        super(fusion_layer, self).__init__()
        self.cat_mode = cat_mode
        self.modes = 32
        if self.cat_mode == 'dim':
            # concate on d_model dim
            self.linear = nn.Linear(configs.d_model * 2, configs.d_model)
        elif self.cat_mode == 'seq':
            self.linear = nn.Linear(configs.seq_len * 2, configs.seq_len)
        elif self.cat_mode == 'seq_c':
            if configs.seq_len % (2**configs.e_layers) == 0:
                factor = 0
            else:
                factor = 1
            seq_len = configs.seq_len//(2**configs.e_layers)+factor
            self.linear = nn.Linear(seq_len * 2, seq_len)
        elif self.cat_mode == 'weight_sum':
            self.weighted_sum = WeightedSum(configs.seq_len, configs.d_model)
        elif self.cat_mode == 'former':
            # Decoder
            if attn == 'full':
                Attn = FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False)
            elif attn == 'prob':
                Attn = ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False)
            elif attn == 'dsa':
                Attn = DSAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False)
            elif attn == 'fourier':
                Attn = FourierCrossAttention(in_channels=configs.d_model,
                                      out_channels=configs.d_model,
                                      seq_len_q=configs.seq_len,
                                      seq_len_kv=configs.seq_len,
                                      modes=self.modes,
                                      mode_select_method='random')
            elif attn == 'wavelet':
                Attn = MultiWaveletCross(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=configs.seq_len ,
                                                  seq_len_kv=configs.seq_len,
                                                  modes=self.modes,
                                                  ich=configs.d_model,
                                                  base='legendre',
                                                  activation='tanh')
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            Attn,
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            Attn,
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),

            )

    def forward(self, x1, x2):
        if self.cat_mode == 'dim':
            out = torch.cat((x1, x2), dim=2)
            out = self.linear(out)
        elif self.cat_mode == 'seq' or self.cat_mode == 'seq_c':
            out = torch.cat((x1, x2), dim=1)
            out = out.permute(0, 2, 1)
            out = self.linear(out)
            out = out.permute(0, 2, 1)
        elif self.cat_mode == 'weight_sum':
            out = self.weighted_sum(x1, x2)
        elif self.cat_mode == 'none':
            out = x1 + x2
        elif self.cat_mode == 'former':
            out = self.decoder(x1, x2, x_mask=None, cross_mask=None)
        return out

class ConvLayer_m(nn.Module):
    def __init__(self, in_c, out_c, conv='down'):
        super(ConvLayer_m, self).__init__()
        i = 2
        kernel_size = 2*i+1
        padding = i if torch.__version__ >= '1.5.0' else 2*i
        if conv == 'down':
            self.Conv = nn.Conv1d(in_channels=in_c,
                                  out_channels=out_c,
                                  kernel_size=kernel_size,
                                  stride=2,
                                  padding=padding,
                                  padding_mode='circular')
        else:
            self.Conv = nn.ConvTranspose1d(in_channels=in_c,
                                           out_channels=out_c,
                                           kernel_size=kernel_size,
                                           stride=2,
                                           padding=padding,
                                           padding_mode='zeros')

        self.activation = nn.GELU()

    def forward(self, x):
        # x:[B,S,C]
        x = x.permute(0, 2, 1)  # [B,C,S]
        x = self.Conv(x)
        x = x.permute(0, 2, 1)  # [B,S,C]
        x = self.activation(x)
        return x