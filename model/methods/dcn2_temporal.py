from model.methods.base_temporal import Method_Temporal

class DCN2_TemporalMethod(Method_Temporal):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert args.cat_policy == 'indices'
        assert args.enable_timestamp, "Temporal method requires timestamp"
        assert args.temporal_policy == 'indices'

    def construct_model(self, model_config = None):
        if model_config is None:
            model_config = self.args.config['model']
        from model.models.dcn2_temporal import DCNv2_Temporal
        self.model = DCNv2_Temporal(
            categories=self.categories,
            d_in=self.d_in,
            d_out=self.d_out,
            t_mean = self.args.t_mean,
            t_std = self.args.t_std,
            **model_config
        ).to(self.args.device)