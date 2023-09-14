import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.block import Block_1d, PermuteLayer
from timm.models.layers import trunc_normal_, DropPath


class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers # stem先来降低最大的一次
        stem = nn.Sequential(
            nn.Conv1d(in_chans, dims[0], kernel_size=4, stride=4),  # [B,C,T]
            PermuteLayer((0, 2, 1)),  # [B,T,C]
            nn.LayerNorm(dims[0]),
            PermuteLayer((0, 2, 1)),
        )
        self.downsample_layers.append(stem)
        # 这里只是三层，每次先执行降采样然后才是进入到stage里面去
        for i in range(3):
            downsample_layer = nn.Sequential(
                PermuteLayer((0, 2, 1)),  # [B,T,C]
                nn.LayerNorm(dims[i]),
                PermuteLayer((0, 2, 1)),  # [B,C,T]
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        # 这个rate的插补是全局进行的，从最初层到最后的层，沿着depth方向逐渐增加
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        # 这个地方值得学习，正常nn.sequential里面是不能放list的，必须是一堆layer，这样的话就可以使用list的形式来写了
        # 使用nn.sequential的好处是不用写forward函数
        # 一共四个stage，每个stage的block的数目受depth控制
        for i in range(4):
            stage = nn.Sequential(
                *[Block_1d(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1])  # final norm layer


        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x  # global average pooling, (N, C, H, W) -> (N, C) # 同时沿着两个维度上面去取mean

    def forward(self, x):
        x = self.forward_features(x)
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
        self.model = ConvNeXtV2(depths=[1*configs.e_layers, 1*configs.e_layers, 3*configs.e_layers, 1*configs.e_layers],
                                dims=[configs.d_model, configs.d_model*2, configs.d_model*2, configs.d_model*8],
                                drop_path_rate=configs.dropout,
                                in_chans=configs.d_model)
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
            self.projection = nn.Linear(
                configs.d_model * 8, configs.num_class)

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
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        enc_out = enc_out.permute(0, 2, 1)  # [B,C,T]
        enc_out = self.model(enc_out)
        enc_out = enc_out.permute(0, 2, 1)  # [B,T,C]
        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        # output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.mean(dim=1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # 从后往前数的那部分才是输入，原本完整的dec_out:[batch_size,label+pred,dim]
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
