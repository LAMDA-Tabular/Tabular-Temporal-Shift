import numpy as np
import os

THIS_PATH = os.path.dirname(os.path.abspath(__file__))

def split_data(seed, train_idx, val_idx, option='none', ratio=0.5):
    train_size = len(train_idx)
    val_size = len(val_idx)
    combined_idx = np.concatenate((train_idx, val_idx))
    np.random.seed(seed)
    if option == 'none':
        np.random.shuffle(combined_idx)
        new_train_idx = combined_idx[:train_size]
        new_val_idx = combined_idx[train_size:]
    elif option == 'last':
        val_in_last = int(val_size * ratio)
        val_in_other = val_size - val_in_last
        last_idx = combined_idx[-val_size:]
        other_idx = combined_idx[:-val_size]
        np.random.shuffle(last_idx)
        np.random.shuffle(other_idx)
        new_train_idx = np.concatenate((other_idx[val_in_other:], last_idx[val_in_last:]))
        new_val_idx = np.concatenate((other_idx[:val_in_other], last_idx[:val_in_last]))
    elif option == 'foremost':
        val_in_foremost = int(val_size * ratio)
        val_in_other = val_size - val_in_foremost
        foremost_idx = combined_idx[:val_size]
        other_idx = combined_idx[val_size:]
        np.random.shuffle(foremost_idx)
        np.random.shuffle(other_idx)
        new_train_idx = np.concatenate((foremost_idx[val_in_foremost:], other_idx[val_in_other:]))
        new_val_idx = np.concatenate((foremost_idx[:val_in_foremost], other_idx[:val_in_other]))
        
    return new_train_idx, new_val_idx


def save_split(dataset, seed, train_idx, val_idx, test_idx, option='none'):
    if option == 'none':
        split_name = f'holdout_random_{seed}'
    elif option == 'last':
        split_name = f'holdout_random_{seed}_last'
    elif option == 'foremost':
        split_name = f'holdout_random_{seed}_foremost'
    save_path = os.path.join(THIS_PATH, dataset, split_name)
    os.makedirs(save_path, exist_ok=True)
    new_train_idx, new_val_idx = split_data(seed, train_idx, val_idx, option)
    print(f'dataset: {dataset}, seed: {seed}, option: {option}')
    print(f'train: {len(new_train_idx)}, val: {len(new_val_idx)}, test: {len(test_idx)}')
    np.save(os.path.join(save_path, f"train_idx.npy"), new_train_idx)
    np.save(os.path.join(save_path, f"val_idx.npy"), new_val_idx)
    np.save(os.path.join(save_path, f"test_idx.npy"), test_idx)


def process_datasets(datasets):
    for dataset in datasets:
        base_path = os.path.join(THIS_PATH, '..', '..', 'data_tabred', dataset, 'split-default')
        train_idx = np.load(os.path.join(base_path, "train_idx.npy"))
        val_idx = np.load(os.path.join(base_path, "val_idx.npy"))
        test_idx = np.load(os.path.join(base_path, "test_idx.npy"))
        for seed in range(3):
            save_split(dataset, seed, train_idx, val_idx, test_idx)
            # save_split(dataset, seed, train_idx, val_idx, test_idx, option='last')
            # save_split(dataset, seed, train_idx, val_idx, test_idx, option='foremost')


datasets = ['cooking-time', 'delivery-eta', 'ecom-offers', 'homecredit-default', 'homesite-insurance', 'maps-routing', 'sberbank-housing', 'weather']
process_datasets(datasets)