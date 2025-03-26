from model.methods.base import Method
import time
import torch
import os.path as osp
from tqdm import tqdm
import numpy as np
from model.utils import (
    Averager
)

from model.lib.data import (
    Dataset_TS,
    data_nan_process,
    data_enc_process,
    num_enc_process,
    data_norm_process,
    data_label_process,
    data_loader_process_TS,
    get_categories
)

class Method_Temporal(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert args.enable_timestamp, "requires timestamp"


    def data_format(self, is_train = True, N = None, C = None, M = None, y = None):
        """
        Format the data for training or testing.

        :param is_train: bool, whether the data is for training or testing
        :param N: dict, numerical data
        :param C: dict, categorical data
        :param y: dict, labels
        """
        if is_train:
            self.N, self.C, self.num_new_value, self.imputer, self.cat_new_value = data_nan_process(self.N, self.C, self.args.num_nan_policy, self.args.cat_nan_policy)
            self.y, self.y_info, self.label_encoder = data_label_process(self.y, self.is_regression)
            self.N,self.num_encoder = num_enc_process(self.N,num_policy = self.args.num_policy, n_bins = self.args.config['training']['n_bins'],y_train=self.y['train'],is_regression=self.is_regression)
            self.N, self.C, self.ord_encoder, self.mode_values, self.cat_encoder = data_enc_process(self.N, self.C, self.args.cat_policy, self.y['train'])
            self.N, self.normalizer = data_norm_process(self.N, self.args.normalization, self.args.seed)
            
            if self.is_regression:
                self.d_out = 1
            else:
                self.d_out = len(np.unique(self.y['train']))
            self.d_in = 0 if self.N is None else self.N['train'].shape[1]
            self.categories = get_categories(self.C)
            self.N, self.C, self.M, self.y, self.train_loader, self.val_loader, self.criterion = data_loader_process_TS(self.is_regression, (self.N, self.C), self.M, self.y, self.y_info, self.args.device, self.args.batch_size, is_train = True)

        else:
            N_test, C_test, _, _, _ = data_nan_process(N, C, self.args.num_nan_policy, self.args.cat_nan_policy, self.num_new_value, self.imputer, self.cat_new_value)
            y_test, _, _ = data_label_process(y, self.is_regression, self.y_info, self.label_encoder)
            N_test, _ = num_enc_process(N_test, num_policy=self.args.num_policy, n_bins=self.args.config['training']['n_bins'],y_train=None, encoder=self.num_encoder)
            N_test, C_test, _, _, _ = data_enc_process(N_test, C_test, self.args.cat_policy, None, self.ord_encoder, self.mode_values, self.cat_encoder)
            N_test, _ = data_norm_process(N_test, self.args.normalization, self.args.seed, self.normalizer)
            _, _, _, _, self.test_loader, _ =  data_loader_process_TS(self.is_regression, (N_test, C_test), M, y_test, self.y_info, self.args.device, self.args.batch_size, is_train = False)
            if N_test is not None and C_test is not None:
                self.N_test, self.C_test = N_test['test'], C_test['test']
            elif N_test is None and C_test is not None:
                self.N_test, self.C_test = None, C_test['test']
            else:
                self.N_test, self.C_test = N_test['test'], None
            self.M_test = M['test']
            self.y_test = y_test['test']
    
    
    def fit(self, data, info, train = True, config = None, best_epoch = None):
        """
        Fit the method to the data.

        :param data: tuple, (N, C, y)
        :param info: dict, information about the data
        :param train: bool, whether to train the method
        :param config: dict, configuration for the method
        :return: float, time cost
        """
        N, C, M, y = data
        # if the method already fit the dataset, skip these steps (such as the hyper-tune process)
        self.D = Dataset_TS(N=N, C=C, M=M, y=y, info=info)
        self.N, self.C, self.M, self.y = self.D.N, self.D.C, self.D.M, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
        self.n_num_features, self.n_cat_features = self.D.n_num_features, self.D.n_cat_features
        if config is not None:
            self.reset_stats_withconfig(config)
        self.data_format(is_train = True)
        self.args.t_mean = self.D.t_mean
        self.args.t_std = self.D.t_std
        self.construct_model()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.args.config['training']['lr'], 
            weight_decay=self.args.config['training']['weight_decay']
        )
        # if not train, skip the training process. such as load the checkpoint and directly predict the results
        if not train:
            return

        time_cost = 0
        
        if best_epoch is None:
            for epoch in range(self.args.max_epoch):
                tic = time.time()
                self.train_epoch(epoch)
                self.validate(epoch)
                elapsed = time.time() - tic
                time_cost += elapsed
                print(f'Epoch: {epoch}, Time cost: {elapsed}')
                if not self.continue_training:
                    break
            torch.save(
                dict(params=self.model.state_dict()),
                osp.join(self.args.save_path, 'epoch-last-{}.pth'.format(str(self.args.seed)))
            )
        else:
            for epoch in range(best_epoch + 1):
                tic = time.time()
                self.train_epoch(epoch)
                elapsed = time.time() - tic
                time_cost += elapsed
                print(f'Epoch: {epoch}, Time cost: {elapsed}')
            torch.save(
                dict(params=self.model.state_dict()),
                osp.join(self.args.save_path, 'best-val-{}.pth'.format(str(self.args.seed)))
            )
        return time_cost


    def predict(self, data, info, model_name):
        """
        Predict the results of the data.

        :param data: tuple, (N, C, y)
        :param info: dict, information about the data
        :param model_name: str, name of the model
        :return: tuple, (loss, metric, metric_name, predictions)
        """
        N, C, M, y = data
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, model_name + '-{}.pth'.format(str(self.args.seed))))['params'])
        print('best epoch {}, best val res={:.4f}'.format(self.trlog['best_epoch'], self.trlog['best_res']))
        ## Evaluation Stage
        self.model.eval()

        self.data_format(False, N, C, M, y)

        test_logit, test_label = [], []
        with torch.no_grad():
            for i, (X, M, y) in tqdm(enumerate(self.test_loader)):
                if self.N is not None and self.C is not None:
                    X_num, X_cat = X[0], X[1]
                elif self.C is not None and self.N is None:
                    X_num, X_cat = None, X
                else:
                    X_num, X_cat = X, None  
                        
                pred = self.model(X_num, X_cat, M)

                test_logit.append(pred)
                test_label.append(y)
                
        test_logit = torch.cat(test_logit, 0)
        test_label = torch.cat(test_label, 0)
        
        vl = self.criterion(test_logit, test_label).item()     

        vres, metric_name = self.metric(test_logit, test_label, self.y_info)

        print('Test: loss={:.4f}'.format(vl))
        for name, res in zip(metric_name, vres):
            print('[{}]={:.4f}'.format(name, res))
        
        return vl, vres, metric_name, test_logit


    def train_epoch(self, epoch):
        """
        Train the model for one epoch.

        :param epoch: int, the current epoch
        """
        self.model.train()
        tl = Averager()
        for i, (X, M, y) in enumerate(self.train_loader, 1):
            self.train_step = self.train_step + 1
            if self.N is not None and self.C is not None:
                X_num, X_cat = X[0], X[1]
            elif self.C is not None and self.N is None:
                X_num, X_cat = None, X
            else:
                X_num, X_cat = X, None

            loss = self.criterion(self.model(X_num, X_cat, M), y)

            tl.add(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if (i-1) % 50 == 0 or i == len(self.train_loader):
                print('epoch {}, train {}/{}, loss={:.4f} lr={:.4g}'.format(
                    epoch, i, len(self.train_loader), loss.item(), self.optimizer.param_groups[0]['lr']))
            del loss
        tl = tl.item()
        self.trlog['train_loss'].append(tl)    


    def validate(self, epoch):
        """
        Validate the model.

        :param epoch: int, the current epoch
        """
        print('best epoch {}, best val res={:.4f}'.format(
            self.trlog['best_epoch'], 
            self.trlog['best_res']))
        
        ## Evaluation Stage
        self.model.eval()
        test_logit, test_label = [], []
        with torch.no_grad():
            for i, (X, M, y) in tqdm(enumerate(self.val_loader)):
                if self.N is not None and self.C is not None:
                    X_num, X_cat = X[0], X[1]
                elif self.C is not None and self.N is None:
                    X_num, X_cat = None, X
                else:
                    X_num, X_cat = X, None                            

                pred = self.model(X_num, X_cat, M)

                test_logit.append(pred)
                test_label.append(y)
                
        test_logit = torch.cat(test_logit, 0)
        test_label = torch.cat(test_label, 0)
        
        vl = self.criterion(test_logit, test_label).item()   

        if self.is_regression:
            task_type = 'regression'
            measure = np.less_equal
        else:
            task_type = 'classification'
            measure = np.greater_equal

        vres, metric_name = self.metric(test_logit, test_label, self.y_info)

        print('epoch {}, val, loss={:.4f} {} result={:.4f}'.format(epoch, vl, task_type, vres[0]))
        if measure(vres[0], self.trlog['best_res']) or epoch == 0:
            self.trlog['best_res'] = vres[0]
            self.trlog['best_epoch'] = epoch
            torch.save(
                dict(params=self.model.state_dict()),
                osp.join(self.args.save_path, 'best-val-{}.pth'.format(str(self.args.seed)))
            )
            self.val_count = 0
        else:
            if self.val_count == self.args.early_stopping:
                self.continue_training = False
            self.val_count += 1
        torch.save(self.trlog, osp.join(self.args.save_path, 'trlog'))