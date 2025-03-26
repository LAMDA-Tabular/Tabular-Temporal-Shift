from model.methods.base_temporal import Method_Temporal
import torch
import numpy as np
import os.path as osp
from tqdm import tqdm
from model.utils import (
    Averager,
)
from model.methods.tabm import loss_fn


class TabM_TemporalMethod(Method_Temporal):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert args.cat_policy == 'indices'
        assert args.num_policy == 'none'
        assert args.enable_timestamp, "Temporal method requires timestamp"
        assert args.temporal_policy == 'indices'


    def construct_model(self, model_config = None):
        from model.models.tabm_temporal import TabM_Temporal
        if model_config is None:
            model_config = self.args.config['model']
        self.model = TabM_Temporal(
            n_num_features=self.d_in,
            cat_cardinalities=self.categories,
            n_classes=self.d_out,
            t_mean = self.args.t_mean,
            t_std = self.args.t_std,
            **model_config
        ).to(self.args.device)


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
                pred = pred.mean(1)
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

            loss = loss_fn(self.criterion, self.model(X_num, X_cat, M), y)

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
                pred = pred.mean(1)
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