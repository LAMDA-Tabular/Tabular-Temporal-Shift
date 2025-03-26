from model.methods.modernNCA_temporal import ModernNCA_TemporalMethod


class ModernNCA_Temp_PLRMethod(ModernNCA_TemporalMethod):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)


    def construct_model(self, model_config = None):
        from model.models.modernNCA_temp_plr import ModernNCA_Temp_PLR
        if model_config is None:
            model_config = self.args.config['model']
        self.model = ModernNCA_Temp_PLR(
            d_in = self.n_num_features + self.n_cat_features,
            d_num = self.n_num_features,
            d_out = self.d_out,
            t_mean = self.args.t_mean,
            t_std = self.args.t_std,
            **model_config
        ).to(self.args.device)