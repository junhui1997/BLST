from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from layers.Lion import Lion
from sklearn.metrics import confusion_matrix
import torch.optim.lr_scheduler as lr_scheduler
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pdb

warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)


    def _build_model(self):
        data_dic = {'bone_drill_c': {'num_classes': 3}}
        if self.args.data in data_dic.keys():
            self.args.pred_len = 0
            self.args.num_class = data_dic[self.args.data]['num_classes']
        else:
            # model input depends on data
            train_data, train_loader = self._get_data(flag='TRAIN')
            test_data, test_loader = self._get_data(flag='TEST')
            # enc_in在参数里面直接设置
            self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
            self.args.pred_len = 0
            self.args.enc_in = train_data.feature_df.shape[1]
            self.args.num_class = len(train_data.class_names)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = Lion(self.model.parameters(), lr=self.args.learning_rate, weight_decay=0.1)
        return model_optim

    def _select_scheduler(self, optimizer):
        if self.args.lradj == 'type3':
            # T_max是半个周期的长度，也就是从最大值到最小值需要用的时长
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=3, eta_min=0.0001)
        elif self.args.lradj == 'type4':
            # patience = 2,代表的是3次val 没有下降后开始降低learning rate
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=5)
        else:
            scheduler = None
        return scheduler

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                # print(pred.shape, label.shape)
                loss = criterion(pred, label.long().squeeze(-1).cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        scheduler = self._select_scheduler(model_optim)
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())
                if (i + 1) % 200 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                    .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))


            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # type 1,2时候是每隔固定周期调整一次，type 3,4时候是每个周期调整一次
            if (epoch + 1) % 5 == 0 or self.args.lradj == 'type3' or self.args.lradj == 'type4':
                adjust_learning_rate(model_optim, epoch + 1, self.args, scheduler, vali_loss)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        time_now = time.time()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        # check inference time
        duration = time.time() -time_now
        print('total time ', duration, duration/len(test_data))
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        # print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            print('recording res')
            os.makedirs(folder_path)
        cf_matrix = confusion_matrix(predictions, trues)
        cm_normalized = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
        df = pd.DataFrame(cm_normalized, index=['None', 'CTB', 'CCB'], columns=['None', 'CTB', 'CCB'])
        df.to_pickle(folder_path + 'confusion_m.pkl')
        sns.heatmap(df, annot=True,  cmap="YlGnBu", vmax=1, fmt="g")  # cbar=None,
        # plt.title("Confusion Matrix"), plt.tight_layout()
        # plt.xlabel("True Class"),
        # plt.ylabel("Predicted Class")
        plt.savefig(folder_path + 'confusion_matrix.png')
        plt.clf()

        print('accuracy:{}'.format(accuracy))
        f = open("result_classification.txt", 'a')
        f.write(setting + "  \n")
        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('\n')
        f.close()
        return
