# Understanding the Limits of Deep Tabular Methods with Temporal Shift

This paper is submitted to **ICML'25 (Submission 10139)**, under review.

## Introduction

We adopt training, evaluation and tuning setup from Ye et al [1]. We tune hyper-parameters using Optuna, performing 100 trials for most methods to identify the best configuration. The hyper-parameter search space follows exactly the settings in Rubachev et al [2]. Using these optimal hyper-parameters, each method is trained with 15 random seeds, and the average performance across seeds is reported. For all deep learning methods, we use a batch size of 1024 and
AdamW as the optimizer.

We followed Rubachev et al [2] and performed only 25 hyper-parameter tuning runs for FT-T and TabR, as these methods exhibit lower efficiency on datasets with large feature dimensions and sample sizes. For our temporal embedding, we conducted separate hyper-parameter searches for the periodic order and linear trend. However, since 25 tuning trials were insufficient to identify optimal hyper-parameters for the temporal embedding, we performed a global tuning of the temporal embedding order for FT-T and TabR.

For classification tasks, we evaluate models using AUC (higher is better) as the primary metric and use RMSE (lower is better) for regression tasks to select the best-performing model during training on the validation set.

To ensure the validity of random splitting, each group of random split experiments was tested on three distinct random splits, with 15 random seeds run on each split. The mean performance across these runs is reported as the final result. The variance of the random split is calculated based on all 45 results (3 splits × 15 seeds), as the random split is subject to variance from both the split selection and the running seeds during the training phase. This approach better reflects the overall stability of the standard procedure.

[1] Ye, H.-J., Liu, S.-Y., Cai, H.-R., Zhou, Q.-L., and Zhan, D.-C. A closer look at deep learning methods on tabular datasets. CoRR, abs/2407.00956, 2024.

[2] Rubachev, I., Kartashev, N., Gorishniy, Y., and Babenko, A. Tabred: A benchmark of tabular machine learning in-the-wild. In ICLR, 2025.

## Usage

### Deep method

For deep methods, run:

```bash
python train_model_deep.py --dataset $DATASET_NAME \
						   --enable_timestamp \
						   --validate_option $VAL_OPTION \
						   --model_type $MODEL_NAME \
						   --cat_policy $CAT_POLICY \
						   --temporal_policy $TEMPORAL_POLICY \
						   --gpu 0 --max_epoch 200 --seed_num 15 \
						   --tune --retune --n_trials 100
```

- `DATASET_NAME`: Dataset name in TabReD benchmark.

  ```bash
  choices=(cooking-time, delivery-eta, ecom-offers, homecredit-default,
           homesite-insurance, maps-routing, sberbank-housing, weather)
  ```

- `VAL_OPTION`: Validation set splitting strategy. Random splits are fixed in `data_splits/`.

  ```bash
  choices=(
  	holdout_last,                    # Original splitting strategy in TabReD
  	holdout_foremost_sample,         # Our training protocol
  	holdout_last_nobias_lag_sample,           # Split (a), w/  lag, w/o bias
  	holdout_last_nobias_nolag_sample,         # Split (b), w/o lag, w/o bias
  	holdout_last_bias_lag_sample,             # Split (c), w/  lag, w/  bias
  	holdout_last_nobias_nolag_reverse_sample, # Split (d), w/o lag, w/o bias
  	holdout_random_0,                # Random split 0
      holdout_random_1,                # Random split 1
      holdout_random_2,                # Random split 2
  )
  ```

- `MODEL_NAME`: Deep method name. `*_temporal ` means model with our temporal embedding.

  ```bash
  choices=(
      mlp,       mlp_temporal,
      mlp_plr,   mlp_plr_temporal,
      snn,       snn_temporal,
      dcn2,      dcn2_temporal,
      ftt,       ftt_temporal,
      tabr,      tabr_temporal,
      modernNCA, modernNCA_temporal,
      tabm,      tabm_temporal,
  )
  ```

- `CAT_POLICY`: Categorical feature policy. We fix this policy to one-hot encoding.

  ```bash
  case $method in
      modernNCA*|tabr*) 
          cat_policy=tabr_ohe
          ;;
      mlp_plr*|tabm*|ftt*|dcn2*|snn*)
          cat_policy=indices
          ;;
      *)
          cat_policy=ohe
          ;;
  esac
  ```

- `TEMPORAL_POLICY`: Timestamp policy.

  ```bash
  choices=(
  	indices,           # None in paper
      num,               # Num in paper
      time_num,          # Time in paper
  )
  ```

### Classical method

For classical methods, run:

```bash
python train_model_classical.py --dataset $DATASET_NAME \
                                --enable_timestamp \
                                --validate_option $VAL_OPTION \
                                --model_type $MODEL_NAME \
                                --cat_policy $CAT_POLICY \
                                --gpu "" --seed_num 15 \
                                --tune --retune --n_trials 100
```

- `DATASET_NAME` and `VAL_OPTION` share the same choices with deep methods.

- `MODEL_NAME`: Classical method name.

  ```bash
  choices=(
  	XGBoost, 
  	LightGBM, 
  	CatBoost, 
  	RandomForest, 
  	SGD,           # Linear in paper. TabReD also adopts SGD as linear model.
  )
  ```

- `CAT_POLICY`: Categorical feature policy. We fix this policy to one-hot encoding.

  ```bash
  case $method in
      catboost)
          cat_policy=indices
          ;;
      *)
          cat_policy=ohe
          ;;
  esac
  ```

  

  

  **Enjoy the code!** 