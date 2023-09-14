import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding


class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, conv='down'):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        if conv == 'down':
            self.Conv = nn.Conv1d(in_channels=in_c,
                                  out_channels=out_c,
                                  kernel_size=3,
                                  stride=2,
                                  padding=padding,
                                  padding_mode='circular')
        else:
            self.Conv = nn.ConvTranspose1d(in_channels=in_c,
                                           out_channels=out_c,
                                           kernel_size=3,
                                           stride=2,
                                           padding=padding,
                                           padding_mode='zeros')
        self.norm = nn.BatchNorm1d(out_c)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.Conv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = x.transpose(1, 2)
        return x


class fcn_n(nn.Module):
    def __init__(self, configs):
        super(fcn_n, self).__init__()
        self.e_layer = configs.e_layers
        self.d_model = configs.d_model
        self.seq_len = configs.seq_len
        self.down_sample = nn.ModuleList(
            [ConvLayer(self.d_model * (2 ** i), self.d_model * (2 ** (i + 1)), 'down') for i in
             range(configs.e_layers)])
        self.d_model_f = self.d_model * (2 ** configs.e_layers)
        self.up_sample = nn.ModuleList(
            [ConvLayer(self.d_model_f // (2 ** i), self.d_model_f // (2 ** (i + 1)), 'up') for i in
             range(configs.e_layers)])

    def forward(self, x):
        for i in range(self.e_layer):
            x = self.down_sample[i](x)
        for i in range(self.e_layer):
            x = self.up_sample[i](x)
        padding_type = 'zero'
        padding_type = 'interpolate'
        if padding_type == 'zero':
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # padding_part = torch.zeros([x.shape[0], self.seq_len-x.shape[1], x.shape[2]]).to(device)
            # x = torch.cat((x, padding_part), dim=1)
            x = x.permute(0, 2, 1)
            x = F.pad(x, (0, self.seq_len-x.shape[2]))
            x = x.permute(0, 2, 1)
        elif padding_type == 'interpolate':
            x = x.permute(0, 2, 1)
            x = F.interpolate(x, size=[self.seq_len], mode='linear')
            x = x.permute(0, 2, 1)
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
        self.model = fcn_n(configs)
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.enc_embedding_n = nn.Linear(configs.enc_in, configs.d_model)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
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
