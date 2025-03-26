from tqdm import tqdm
from model.utils import (
    get_classical_args,
    show_results_classical,
    tune_hyper_parameters,
    get_method,
    set_seeds
)
from model.lib.data import (
    get_dataset,
    get_dataset_tabred
)


if __name__ == '__main__':
    results_list, time_list = [], []
    args,default_para,opt_space = get_classical_args()
    if args.enable_timestamp:
        train_val_data, test_data, info = get_dataset_tabred(args)
    else:
        train_val_data, test_data, info = get_dataset(args.dataset,args.dataset_path)
    if args.tune:
        args, best_epoch = tune_hyper_parameters(args, opt_space, train_val_data, info, validate_option=args.validate_option)
    
    ## Training Stage over different random seeds
    for seed in tqdm(range(args.seed_num)):
        args.seed = seed    # update seed  
        set_seeds(args.seed)
        method = get_method(args.model_type)(args, info['task_type'] == 'regression')
        assert args.validate_option.startswith('holdout')
        time_cost = method.fit(train_val_data, info, train=True)    
        vres, metric_name, predict_logits = method.predict(test_data, info, model_name=args.evaluate_option)

        results_list.append(vres)
        time_list.append(time_cost)
    show_results_classical(args,info, metric_name,results_list,time_list)
