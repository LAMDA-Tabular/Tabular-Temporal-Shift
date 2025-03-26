from model.methods.base_temporal import Method_Temporal

class MLP_PLR_TemporalMethod(Method_Temporal):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert args.cat_policy == 'indices'
        assert args.enable_timestamp, "MLP_PLR_TemporalMethod requires timestamp"
        assert args.temporal_policy == 'indices'

    def construct_model(self, model_config = None):
        from model.models.mlp_plr_temporal import MLP_Temporal
        if model_config is None:
            model_config = self.args.config['model']
        self.model = MLP_Temporal(
            d_in=(self.d_in + len(self.categories)) if self.categories is not None else self.d_in,
            d_num=self.d_in,
            d_out=self.d_out,
            t_mean = self.args.t_mean,
            t_std = self.args.t_std,
            **model_config
        ).to(self.args.device)