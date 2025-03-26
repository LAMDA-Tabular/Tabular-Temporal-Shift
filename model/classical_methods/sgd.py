from model.classical_methods.base import classical_methods
import os.path as ops
import pickle
import time

from sklearn.metrics import roc_auc_score, root_mean_squared_error

class SGDMethod(classical_methods):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(args.cat_policy != 'indices')
        self.is_regression = is_regression

    def construct_model(self, model_config = None):
        if model_config is None:
            model_config = self.args.config['model']
        from sklearn.linear_model import SGDClassifier, SGDRegressor
        self.model = SGDRegressor(**model_config, loss='squared_error', random_state=self.args.seed) if self.is_regression else SGDClassifier(**model_config, loss='log_loss', random_state=self.args.seed)
    
    def fit(self, data, info, train=True, config=None):
        super().fit(data, info, train, config)
        # if not train, skip the training process. such as load the checkpoint and directly predict the results
        if not train:
            return
        tic = time.time()
        self.model.fit(self.N['train'], self.y['train'])
        if self.is_regression:
            y_pred_val = self.model.predict(self.N['val'])
            self.trlog['best_res'] = root_mean_squared_error(self.y['val'], y_pred_val)
        else:
            y_pred_val = self.model.predict_proba(self.N['val'])[:,1]
            self.trlog['best_res'] = roc_auc_score(self.y['val'], y_pred_val)
        time_cost = time.time() - tic
        with open(ops.join(self.args.save_path , 'best-val-{}.pkl'.format(self.args.seed)), 'wb') as f:
            pickle.dump(self.model, f)
        return time_cost
        
    
    def predict(self, data, info, model_name):
        if self.args.enable_timestamp:
            N, C, M, y = data
        else:
            N, C, y = data
        with open(ops.join(self.args.save_path , 'best-val-{}.pkl'.format(self.args.seed)), 'rb') as f:
            self.model = pickle.load(f)
        self.data_format(False, N, C, y)
        test_label = self.y_test
        if self.is_regression:
            test_logit = self.model.predict(self.N_test)
        else:
            test_logit = self.model.predict_proba(self.N_test)
        vres, metric_name = self.metric(test_logit, test_label, self.y_info)
        return vres, metric_name, test_logit