# Fair Prescriptive Process Monitoring

Code for training remaining-time predictors on event logs and evaluating their counterfactual fairness with respect to sensitive attributes. It compares a vanilla LSTM baseline with neuro-symbolic models based on Logic Tensor Networks (LTN), and produces next-activity recommendations using a transition system built from the training log.

## Installation

```bash
conda env create -f environment.yml
conda activate fair-prescriptive-pm
```

`main.py` also imports packages that `environment.yml` does not list yet. Install them as well:

```bash
pip install matplotlib seaborn shap dowhy networkx
```

## Data

The event logs are **not included** in this repository. Download them from:

- **BPI Challenge 2013 (incidents)**: https://data.4tu.nl/articles/dataset/BPI_Challenge_2013_incidents/12693914
- **Modified BPI Challenge 2012 and Hospital Billing** (with sensitive attributes): https://zenodo.org/records/15387576

Place them in `data/` using these file names:

| `--dataset`                   | Expected file                          | Sensitive attribute(s) |
|-------------------------------|----------------------------------------|------------------------|
| `bpi12`                       | `data/bpi12.pkl`                       | `gender`               |
| `bpi13`                       | `data/bpi13i.csv`                      | `organization country` |
| `hospital_billing_gender`     | `data/hospital_billing_gender.pkl`     | `gender`               |
| `hospital_billing_age`        | `data/hospital_billing_age.pkl`        | `age`                  |
| `hospital_billing_age_gender` | `data/hospital_billing_age_gender.pkl` | `age`, `gender`        |

Each log needs a case ID column (`case_id` or `case:concept:name`) and an activity column (`activity` or `concept:name`). These five are the only supported values of `--dataset`.

## Usage

Create the output folder for the chosen seed first; the script does not create it:

```bash
mkdir -p run_seed_42
```

```bash
python main.py --train_vanilla --train_nesy --dataset=bpi12 --num_epochs=50 --num_epochs_nesy=50 --seed=42
```

### Arguments

| Argument             | Default    | Description                                                        |
|----------------------|------------|--------------------------------------------------------------------|
| `--dataset`          | `bpi12`    | Dataset to use (see the table above)                               |
| `--train_vanilla`    | off        | Train and evaluate the vanilla LSTM baseline                       |
| `--train_nesy`       | off        | Train and evaluate the LTN models                                  |
| `--num_epochs`       | `50`       | Training epochs for the vanilla LSTM                               |
| `--num_epochs_nesy`  | `50`       | Training epochs for the LTN models                                 |
| `--hidden_size`      | `128`      | LSTM hidden size                                                   |
| `--num_layers`       | `1`        | Number of LSTM layers                                              |
| `--dropout_rate`     | `0.1`      | LSTM dropout rate                                                  |
| `--learning_rate`    | `0.0001`   | Adam learning rate                                                 |
| `--abstraction_mode` | `sequence` | Transition system key abstraction: `sequence` (order matters) or `set` |
| `--seed`             | `42`       | Random seed; also names the `run_seed_<seed>/` output folder       |

With `--train_nesy`, the script trains three LTN variants:

1. **Single stage**: regression and fairness axioms trained together, with Pareto-front monitoring.
2. **Two-stage, stage 1**: regression axioms only.
3. **Two-stage, stage 2**: fine-tuning with regression and fairness axioms, with Pareto-front monitoring.

## Outputs

- `results.txt`: test MAE, RMSE, counterfactual MSE and counterfactual fairness per model (appended to on every run).
- `run_seed_<seed>/<dataset>_<model>_predictions.csv`: test-set predictions per model.
- `<dataset>_<model>_recommended_activities.csv` and `<dataset>_<model>_recommended_activities_counterfactual.csv`: recommended next activities for the original and counterfactual test traces.
- `ltn_pareto_front.pdf`: the Pareto front plot from LTN training. This overwrites the copy committed to the repository.

Preprocessing also writes train/test splits as CSV files (e.g. `data/bpi12_train.csv`).

## Transition system

See [`transition_system/README.MD`](transition_system/README.MD) for how the transition system used for recommendations is built and queried.
