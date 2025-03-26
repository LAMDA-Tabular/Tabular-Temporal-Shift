from model.methods.mlp_plr_temporal import MLP_PLR_TemporalMethod

class MLP_PLR_Temp_PLRMethod(MLP_PLR_TemporalMethod):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)


    def construct_model(self, model_config = None):
        from model.models.mlp_plr_temp_plr import MLP_Temp_PLR
        if model_config is None:
            model_config = self.args.config['model']
        self.model = MLP_Temp_PLR(
            d_in=(self.d_in + len(self.categories)) if self.categories is not None else self.d_in,
            d_num=self.d_in,
            d_out=self.d_out,
            t_mean = self.args.t_mean,
            t_std = self.args.t_std,
            **model_config
        ).to(self.args.device)