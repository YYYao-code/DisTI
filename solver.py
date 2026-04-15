import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from model import Reconstructor

import pickle
from dataloader import get_loader_segment


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss,  model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss

class Solver(object):
    DEFAULTS = {}

    def __init__(self, config,roundpath):

        self.__dict__.update(Solver.DEFAULTS, **config)

        if self.dataset == 'NeurIPSTS':
            print(f'================={self.dataset}_{self.form}======================')
            self.train_loader, label_seq, test_seq = get_loader_segment(batch_size=self.batch_size,
                                                                   seq_length=self.seq_length, form=self.form,
                                                                   step=self.step, mode='train', dataset=self.dataset)
            self.vali_loader, label_seq, test_seq = get_loader_segment(batch_size=self.batch_size,
                                                                  seq_length=self.seq_length, form=self.form,
                                                                  step=self.step, mode='vali', dataset=self.dataset)
            self.test_loader, label_seq, test_seq = get_loader_segment(batch_size=self.batch_size,
                                                                  seq_length=self.seq_length, form=self.form,
                                                                  step=self.step, mode='test', dataset=self.dataset)
            self.thre_loader, label_seq, test_seq = get_loader_segment(batch_size=self.batch_size,
                                                                  seq_length=self.seq_length, form=self.form,
                                                                  step=self.step, mode='thre', dataset=self.dataset)
        else:
            print(f'================={self.dataset}_{self.data_num}======================')
            self.train_loader, label_seq, test_seq = get_loader_segment(batch_size=self.batch_size,
                                                                   seq_length=self.seq_length, form=self.data_num,
                                                                   step=self.step, mode='train', dataset=self.dataset)
            self.vali_loader, label_seq, test_seq = get_loader_segment(batch_size=self.batch_size,
                                                                  seq_length=self.seq_length, form=self.data_num,
                                                                  step=self.step, mode='vali', dataset=self.dataset)
            self.test_loader, label_seq, test_seq = get_loader_segment(batch_size=self.batch_size,
                                                                  seq_length=self.seq_length, form=self.data_num,
                                                                  step=self.step, mode='test', dataset=self.dataset)
            self.thre_loader, label_seq, test_seq = get_loader_segment(batch_size=self.batch_size,
                                                                  seq_length=self.seq_length, form=self.data_num,
                                                                  step=self.step, mode='thre', dataset=self.dataset)

        self.path=roundpath
        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        # self.config=config

    def build_model(self):
        self.model = Reconstructor(c_in=self.c_in,c_out=self.c_out,d_model=self.d_model,window=self.seq_length)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def vali_stage1(self, vali_loader):
        self.model.eval()

        loss = []

        for i, (input_data, _) in enumerate(self.vali_loader):
            input = input_data.float().transpose(1,2).to(self.device)
            output,combined, r_tt, r_tv, r_vv, r_vt, t_feature,v_feature = self.model(input)
            orth_loss = torch.abs(torch.multiply(t_feature, v_feature)).sum(-1).mean()
            dis_loss = self.criterion(r_tt, input) + self.criterion(r_vv, input) + 1 / self.criterion(r_tv,input) + 1 / self.criterion(r_vt, input)

            loss.append((dis_loss+self.w_o*orth_loss).item())

        return np.average(loss)

    def train_stage1(self):

        print("======================TRAIN MODE======================")


        time_now = time.time()


        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)
        epoch_loss=[]
        epoch_times=[]
        for epoch in range(self.num_epochs):
            iter_count = 0
            loss_list = []

            epoch_time = time.time()
            self.model.train()
            for i, input_data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().transpose(1,2).to(self.device)
                output,combined,r_tt,r_tv,r_vv,r_vt,t_feature,v_feature= self.model(input)
                orth_loss=torch.abs(torch.multiply(t_feature,v_feature)).sum(-1).mean()
                dis_loss=self.criterion(r_tt,input)+self.criterion(r_vv,input)+1/self.criterion(r_tv,input)+1/self.criterion(r_vt,input)
                loss = dis_loss+self.w_o*orth_loss
                loss_list.append(loss.item())
                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                loss.backward()
                self.optimizer.step()

            epoch_times.append(time.time() - epoch_time)
            epoch_loss.append(sum(loss_list))
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss_list)
            vali_loss = self.vali_stage1(self.vali_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, self.path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

        train_stats = {'loss': np.asarray(epoch_loss), 'time':np.asarray(epoch_times)}
        with open(self.path + 'train_stats1.pkl', 'wb') as f:
            pickle.dump(train_stats, f)

        return


    def vali_stage2(self, vali_loader):
        self.model.eval()

        loss = []

        for i, (input_data, _) in enumerate(self.vali_loader):

            input = input_data.float().transpose(1,2).to(self.device)
            emb = self.model.transform(input)
            t_feature = self.model.encoder.tconv(emb)
            v_feature = self.model.encoder.vconv(emb.transpose(1, 2)).transpose(1, 2)
            combined1 = torch.multiply(t_feature,nn.Softmax(dim=1)(v_feature)).detach()
            combined2 = torch.multiply(nn.Softmax(dim=-1)(t_feature), v_feature).detach()
            combined = self.model.encoder.merge(torch.concat([combined1,combined2],dim=1))
            output = self.model.encoder.reconstruct(combined.flatten(start_dim=1)).reshape(-1, self.c_in, self.seq_length)
            rec_loss = self.criterion(output, input)
            loss.append(rec_loss.item())

        return np.average(loss)


    def train_stage2(self):

        print("======================TRAIN MODE======================")
        time_now = time.time()
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)
        epoch_loss=[]
        epoch_times=[]
        for epoch in range(self.num2_epochs):
            iter_count = 0
            loss_list = []

            epoch_time = time.time()
            self.model.train()
            for i, input_data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().transpose(1,2).to(self.device)


                emb=self.model.transform(input)
                t_feature = self.model.encoder.tconv(emb)
                v_feature = self.model.encoder.vconv(emb.transpose(1, 2)).transpose(1, 2)
                combined1 = torch.multiply(t_feature, nn.Softmax(dim=1)(v_feature)).detach()
                combined2 = torch.multiply(nn.Softmax(dim=-1)(t_feature), v_feature).detach()
                combined=self.model.encoder.merge(torch.concat([combined1,combined2],dim=1))
                output = self.model.encoder.reconstruct(combined.flatten(start_dim=1)).reshape(-1, self.c_in, self.seq_length)
                rec_loss = self.criterion(output, input)

                loss =  rec_loss
                loss_list.append(loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()


                loss.backward()
                self.optimizer.step()

            epoch_times.append(time.time() - epoch_time)
            epoch_loss.append(sum(loss_list))
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss_list)
            vali_loss = self.vali_stage2(self.vali_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, self.path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

        train_stats = {'loss': np.asarray(epoch_loss), 'time':np.asarray(epoch_times)}
        with open(self.path + 'train_stats2.pkl', 'wb') as f:
            pickle.dump(train_stats, f)

        return



    def test(self):

        self.model.load_state_dict(
            torch.load(
                os.path.join(self.path, str(self.dataset) + '_checkpoint.pth')))

        self.model.eval()

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduction='none')
        test_labels = []
        scores = []
        outputs=[]
        inf_times=[]
        for i, (input_data, labels) in enumerate(self.test_loader):
            inf_start = time.time()
            input = input_data.float().transpose(1,2).to(self.device)
            output,combined, r_tt, r_tv, r_vv, r_vt, t_feature,v_feature = self.model(input)
            inf_times.append(time.time()-inf_start)
            score=criterion(input, output)
            score=torch.mean(score,dim=1)
            test_labels.append(labels.detach().cpu().numpy())
            scores.append(score.detach().cpu().numpy())
            outputs.append(output.detach().cpu().numpy())

        total_inf_time = sum(inf_times)
        per_inf_time = total_inf_time / len(self.test_loader.dataset)

        test_labels = np.concatenate(test_labels, axis=0)

        test_scores = np.concatenate(scores, axis=0)
        test_outputs = np.concatenate(outputs, axis=0) #.reshape(-1)

        result_dict={"test_labels": test_labels, "test_scores": test_scores, 'per_inf_time':per_inf_time}


        file_path = self.path+str(self.data_num)+'_time_evaluation_array.pkl'
        with open(file_path, 'wb') as file_result:
            pickle.dump(result_dict, file_result)

        return