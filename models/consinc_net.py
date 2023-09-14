import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from torch.autograd import Variable
import math
import numpy as np
from layers.block import fusion_layer

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def sinc(band,t_right):
    y_right= torch.sin(2*math.pi*band*t_right)/(2*math.pi*band*t_right)
    y_left= flip(y_right,0)

    y=torch.cat([y_left,Variable(torch.ones(1)).cuda(),y_right])

    return y

# n_filter:这个out_dim
# 后面几个无所谓
# 也是沿着seq_len方向上进行卷积
class sinc_conv(nn.Module):

    def __init__(self, N_filt, Filt_dim, fs):
        super(sinc_conv, self).__init__()

        # Mel Initialization of the filterbanks
        low_freq_mel = 80
        high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, N_filt)  # Equally spaced in Mel scale
        f_cos = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
        b1 = np.roll(f_cos, 1)
        b2 = np.roll(f_cos, -1)
        b1[0] = 30
        b2[-1] = (fs / 2) - 100

        self.freq_scale = fs * 1.0
        self.filt_b1 = nn.Parameter(torch.from_numpy(b1 / self.freq_scale))
        self.filt_band = nn.Parameter(torch.from_numpy((b2 - b1) / self.freq_scale))

        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs

    def forward(self, x):
        filters = Variable(torch.zeros((self.N_filt, self.Filt_dim))).cuda()
        N = self.Filt_dim
        t_right = Variable(torch.linspace(1, (N - 1) / 2, steps=int((N - 1) / 2)) / self.fs).cuda()

        min_freq = 50.0;
        min_band = 50.0;

        filt_beg_freq = torch.abs(self.filt_b1) + min_freq / self.freq_scale
        filt_end_freq = filt_beg_freq + (torch.abs(self.filt_band) + min_band / self.freq_scale)

        n = torch.linspace(0, N, steps=N)

        # Filter window (hamming)
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * n / N);
        window = Variable(window.float().cuda())

        for i in range(self.N_filt):
            low_pass1 = 2 * filt_beg_freq[i].float() * sinc(filt_beg_freq[i].float() * self.freq_scale, t_right)
            low_pass2 = 2 * filt_end_freq[i].float() * sinc(filt_end_freq[i].float() * self.freq_scale, t_right)
            band_pass = (low_pass2 - low_pass1)

            band_pass = band_pass / torch.max(band_pass)

            filters[i, :] = band_pass.cuda() * window

        batch_size, d_model, seq_len = x.shape
        filters = filters.unsqueeze(1)
        filters = filters.repeat(1, d_model, 1)
        # out = F.conv1d(x, filters.view(self.N_filt, 1, self.Filt_dim), padding=padding)  # filter后面的是weight
        out = F.conv1d(x, filters)
        out = F.interpolate(out, size=[seq_len], mode='linear')

        return out
class ConvLayer(nn.Module):
    def __init__(self, configs=None):
        super(ConvLayer, self).__init__()
        # 这里应该是由于api版本的不同，想要维持卷积完后的形状不同需要进行的操作
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # 保持channel的维度不变,此时的cin是d_model
        self.downConv = sinc_conv(configs.d_model, 7, 3)
        self.norm = nn.BatchNorm1d(configs.d_model)
        # 和relu有细微差别
        self.activation = nn.ReLU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)


    # input[batch_size,seq_len,dim]
    def forward(self, x):
        # 这一步同样是为了对d_model进行卷积
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        # 交换回来第一步卷积交换过去的数值
        x = x.transpose(1, 2)
        return x

class ConvLayer_normal(nn.Module):
    def __init__(self, configs=None):
        super(ConvLayer_normal, self).__init__()
        # 这里应该是由于api版本的不同，想要维持卷积完后的形状不同需要进行的操作
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # 保持channel的维度不变,此时的cin是d_model
        self.downConv = nn.Conv1d(in_channels=configs.d_model,
                                  out_channels=configs.d_model,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(configs.d_model)
        # 和relu有细微差别
        self.activation = nn.ReLU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)


    def forward(self, x):
        # 这一步同样是为了对d_model进行卷积
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        # 交换回来第一步卷积交换过去的数值
        x = x.transpose(1, 2)
        return x


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        # e_layer这里是2，注意这里是用for的写法，所以time_block不是共享的参数
        self.model = nn.ModuleList([ConvLayer(configs)
                                    for _ in range(configs.e_layers)])
        self.model2 = nn.ModuleList([ConvLayer_normal(configs)
                                    for _ in range(configs.e_layers)])
        self.fusion = fusion_layer(configs, 'seq_c')
        # embed：timeF， freq:'h'按小时进行的embedding, 这里的d_model没有按照公式上面进行计算，同时需要注意这个d_model特别小，不是512
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.enc_embedding_n = nn.Linear(configs.enc_in, configs.d_model)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # 只有预测任务有predict_linear
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            # bias这个bool量决定是否引入偏置项
            # projection是最后的输出层了
            self.projection = nn.Linear(
                configs.d_model//(2**configs.e_layers), configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            if configs.seq_len % (2**configs.e_layers) == 0:
                factor = 0
            else:
                factor = 1
            self.projection = nn.Linear(
                configs.d_model * (configs.seq_len//(2**configs.e_layers)+factor), configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        print(' do not support this task')
        return 0

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        print(' do not support this task')
        return 0

    def anomaly_detection(self, x_enc):
        print(' do not support this task')
        return 0

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_ins = enc_inc = self.enc_embedding_n(x_enc)  # [B,T,C]
        # enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        for i in range(self.layer):
            enc_inc = self.model[i](enc_inc)
        for i in range(self.layer):
            enc_ins = self.model[i](enc_ins)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        enc_out = self.fusion(enc_inc,enc_ins)
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        # output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            #从后往前数的那部分才是输入，原本完整的dec_out:[batch_size,label+pred,dim]
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
