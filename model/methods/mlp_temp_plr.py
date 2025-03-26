from model.methods.mlp_temporal import MLP_TemporalMethod

class MLP_Temp_PLRMethod(MLP_TemporalMethod):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        

    def construct_model(self, model_config = None):
        from model.models.mlp_temp_plr import MLP_Temp_PLR
        if model_config is None:
            model_config = self.args.config['model']
        self.model = MLP_Temp_PLR(
            d_in=self.d_in,
            d_out=self.d_out,
            t_mean = self.args.t_mean,
            t_std = self.args.t_std,
            **model_config
        ).to(self.args.device)