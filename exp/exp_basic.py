import os
import torch
from torchinfo import summary as summary_info
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, lstm, conv_net, bp, lstm_fcn, fcn, fcn_m, convnext1d, my, sinc_net,consinc_net


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'lstm': lstm,
            'conv_net': conv_net,
            'bp': bp,
            'lstm_fcn': lstm_fcn,
            'fcn': fcn,
            'fcn_m': fcn_m,
            'conv_next': convnext1d,
            'my': my,
            'sinc_net':sinc_net,
            'consinc_net': consinc_net


        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        show_para = 0
        if args.task_name == 'classification' and show_para:
            # 这里快速写出来的方法是直接看一看正常输入时候的shape，根据这个shape去生成不同的input，input最后是写在一个元祖里面去的，但是有一点点不同就是为了计算最后所占用的容量，不能用None，所以需要随机给生成一些
            inputs = (torch.randn(args.batch_size, args.seq_len, args.enc_in), torch.randn(args.batch_size, args.seq_len), torch.randn(args.batch_size, args.seq_len, args.enc_in), torch.randn(args.batch_size, args.seq_len))
            summary_info(
                self.model,
                input_data= inputs,
                col_names=["output_size", "num_params"],
            )
            useless = 0

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
