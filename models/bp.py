import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1

"""
input:【batch_size,seq_len,dim]
output:[batch_size,seq_len,dim]
对于本例input和output都是d_model
nn.linear是针对最后一个dim来做的
"""
class bp_n(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""

    def __init__(self, configs=None):
        super().__init__()
        self.layer = configs.e_layers
        self.bp = nn.ModuleList([nn.Linear(configs.d_model, configs.d_model) for _ in range(self.layer)])

    def forward(self, x):
        for i in range(self.layer):
            x = self.bp[i](x)
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
        self.model = bp_n(configs)
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
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # seq_len -> seq_len+prediction
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        enc_out = self.model(enc_out)
        # porject back
        dec_out = self.projection(enc_out)


        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        print(' do not support this task')
        return 0

    def anomaly_detection(self, x_enc):
        print(' do not support this task')
        return 0

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding_n(x_enc)  # [B,T,C]
        # enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        enc_out = self.model(enc_out)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
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
