# Fair Prescriptive Process Monitoring

This repository trains models that predict the **remaining time** of running process cases and use those predictions to **recommend the next activity**. It then measures whether the recommendations are **counterfactually fair** with respect to sensitive attributes such as gender, age or country.

It compares two kinds of model:

- **Vanilla LSTM**: a remaining-time regressor trained with MSE loss.
- **Neuro-symbolic LTN models**: the same LSTM wrapped in a [Logic Tensor Network](https://github.com/tommasocarraro/LTNtorch). Training maximises how well two kinds of axioms are satisfied: regression axioms (prediction ≈ ground truth) and fairness axioms (prediction for a case ≈ prediction for its counterfactual).

## How it works

1. **Load and split.** `main.py` reads the event log, computes each event's remaining time (in days), and splits cases by case ID order into train, validation and test sets (64% / 16% / 20%).
2. **Transition system.** `transition_system/hash_maps.py` builds a map from training-log prefixes to the activities observed next. These are the candidate activities for recommendation.
3. **Counterfactual log.** `causal_graph_discovery.py` learns a causal graph over the case attributes with FCI (causal-learn). `counterfactuals.py` fits a DoWhy structural causal model on that graph, intervenes on the sensitive attribute, and propagates the change to the attributes that depend on it. The result is a counterfactual copy of every case.
4. **Prefix encoding.** `data/preprocess_*.py` turns each case into padded prefixes. Categorical features are embedded, and the numerical and time features (`time_delta`, `time_of_day`, `day_of_week`) are scaled. Each prefix is paired with its counterfactual version.
5. **Training.** The vanilla LSTM and/or the LTN variants are trained (see [Models](#models)).
6. **Recommendation and fairness.** For each test prefix, the transition system lists the possible next activities. Each one is appended to the prefix with a simulated timestamp (the average execution time of that activity), and the model predicts the resulting remaining time. The activity with the lowest prediction is recommended. This is done for both the original and the counterfactual prefix. **Counterfactual fairness** is the share of test prefixes where both get the same recommendation.

## Repository structure

```
main.py                         Entry point: preprocessing, training, evaluation, recommendation
model/lstm.py                   LSTM regressor with per-feature embeddings
data/
  preprocess_bpi12.py           Preprocessing for BPI12
  preprocess_bpi13.py           Preprocessing for BPI13
  preprocess_hospital_billing_log.py   Preprocessing for the Hospital Billing variants
  dataset.py                    PyTorch datasets (standard and original/counterfactual pairs)
  utils.py                      Remaining time, temporal split, prefix creation, metrics
causal_graph_discovery.py       Causal discovery over case attributes (PC / FCI)
counterfactuals.py              Counterfactual event log generation (DoWhy GCM)
utils.py                        Next-activity recommendation and counterfactual fairness
transition_system/              Transition system (prefix -> next activities), see its README
prosit-recommendations-simulator/   Notebook and simulator to replay recommendations on a BPI12 Petri net
```

## Installation

Use Python 3.11 and install the dependencies with pip:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch ltn pm4py causal-learn dowhy networkx numpy pandas scipy scikit-learn matplotlib seaborn shap catboost pyarrow
```

The `condaenv.*.requirements.txt` files in the repository list only part of these packages.

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
| `hospital_billing_age`        | `data/hospital_billing_age.pkl`        | `age` (binarised: < 65 / ≥ 65) |
| `hospital_billing_age_gender` | `data/hospital_billing_age_gender.pkl` | `age`, `gender`        |

These five are the only supported values of `--dataset`; any other value fails. Each log needs:

- a case ID column: `case_id` or `case:concept:name`
- an activity column: `activity` or `concept:name`
- a `time:timestamp` column

For the Hospital Billing logs, cases with two or fewer events and cases with gender `non conforming` are dropped.

## Usage

Create the output folder for the chosen seed first; the script does not create it:

```bash
mkdir -p run_seed_42
```

Train and evaluate all models on BPI12:

```bash
python main.py --train_vanilla --train_nesy --dataset=bpi12 --num_epochs=50 --num_epochs_nesy=50 --seed=42
```

Run several seeds by repeating the command with a different `--seed`, creating `run_seed_<seed>/` for each.

### Arguments

| Argument             | Default    | Description                                                            |
|----------------------|------------|------------------------------------------------------------------------|
| `--dataset`          | `bpi12`    | Dataset to use (see the table above)                                   |
| `--train_vanilla`    | off        | Train and evaluate the vanilla LSTM baseline                           |
| `--train_nesy`       | off        | Train and evaluate the LTN models                                      |
| `--num_epochs`       | `50`       | Maximum training epochs for the vanilla LSTM                           |
| `--num_epochs_nesy`  | `50`       | Maximum training epochs for each LTN training phase                    |
| `--hidden_size`      | `128`      | LSTM hidden size                                                       |
| `--num_layers`       | `1`        | Number of LSTM layers                                                  |
| `--dropout_rate`     | `0.1`      | LSTM dropout rate                                                      |
| `--learning_rate`    | `0.0001`   | Adam learning rate                                                     |
| `--abstraction_mode` | `sequence` | Transition system key: `sequence` (order matters) or `set` (order ignored) |
| `--seed`             | `42`       | Random seed; also names the `run_seed_<seed>/` output folder           |

Batch size is fixed at 32.

## Models

| Model                        | Enabled by        | Training                                                                                                   | Early stopping / selection |
|------------------------------|-------------------|------------------------------------------------------------------------------------------------------------|----------------------------|
| Vanilla LSTM                 | `--train_vanilla` | MSE loss                                                                                                   | Validation loss, patience 15 |
| LTN, single stage            | `--train_nesy`    | Regression + fairness axioms together                                                                      | Validation Pareto front over the two satisfaction levels, patience 10; keeps the most balanced model |
| LTN, two-stage: stage 1      | `--train_nesy`    | Regression axioms only                                                                                     | Validation, patience 10 |
| LTN, two-stage: stage 2      | `--train_nesy`    | Fine-tunes the stage 1 model on the fairness axiom only                                                    | Validation Pareto front, patience 10 |

Each model is evaluated on the test set for MAE, RMSE, counterfactual MSE (between original and counterfactual predictions) and counterfactual fairness of the recommendations.

## Outputs

All files are written to the current working directory unless noted.

| File                                                         | Contents |
|--------------------------------------------------------------|----------|
| `results.txt`                                                | Test MAE, RMSE, counterfactual MSE and counterfactual fairness per model. Appended to on every run. |
| `run_seed_<seed>/<dataset>_<model>_predictions.csv`          | Test-set predictions per model |
| `<dataset>_<model>_recommended_activities.csv`               | Recommended next activity and predicted remaining time for each original test prefix |
| `<dataset>_<model>_recommended_activities_counterfactual.csv`| The same for the counterfactual prefixes |
| `ltn_pareto_front.pdf`                                       | Validation satisfaction levels and Pareto front from LTN training (overwrites the copy in the repository) |
| `best_model.pt`, `best_model_nesy_regr.pt`, `*_best_pareto_model_e<epoch>_sats[...].pth` | Model checkpoints |
| `data/bpi12_*.csv`, `bpi13_*.csv`, `hospital_billing_age_*.csv` | Train/test splits and preprocessed logs written during preprocessing |
| `simple_test_fci.png`                                        | The discovered causal graph |

## Simulating recommendations

`prosit-recommendations-simulator/RecommendationsSimulator.ipynb` discovers a Petri net and simulation parameters from the BPI12 log, then replays the `*_recommended_activities*.csv` files to estimate how the recommendations change case durations. It contains absolute paths from the authors' machine (`/home/padela/...`) and expects `bpi12a.xes` one level above the notebook. Update these before running it.

## Known issues

These are bugs in the current code, not configuration problems:

- **The recommendation step raises a `NameError`.** `recommend_act_loader` in `utils.py` calls `create_new_feat_vec`, which has been renamed to `create_single_new_feat_vec`.
- **The recommendation step also raises a `TypeError`.** Every `recommend_act_loader(...)` call in `main.py` passes `args.seed` as an extra positional argument, so `save_to_file` is given twice.
- **`transition_system/ts_creation.py`** changes into a hard-coded directory (`/home/padela/Desktop/...`) and has to be edited before it can run.

## Transition system

See [`transition_system/README.MD`](transition_system/README.MD) for how the transition system is built, queried and saved.

## Citation

If you use this code, please cite:

> F. De Santis, A. Padella, G. Park, F. Zanichelli, M. de Leoni. *Fairness in Prescriptive Process Analytics: Integrating Fairness Constraints into the Learning Phase.* In: Business Process Management (BPM). Springer Nature Switzerland, Cham, 2027, pp. 369–387. https://doi.org/10.1007/978-3-032-37777-7_23

```bibtex
@InProceedings{10.1007/978-3-032-37777-7_23,
  author    = {De Santis, Fabrizio and Padella, Alessandro and Park, Gyunam and Zanichelli, Francesco and de Leoni, Massimiliano},
  editor    = {Maggi, Fabrizio Maria and Leopold, Henrik and Santoro, Flavia and Gal, Avigdor},
  title     = {Fairness in Prescriptive Process Analytics: Integrating Fairness Constraints into the Learning Phase},
  booktitle = {Business Process Management},
  year      = {2027},
  publisher = {Springer Nature Switzerland},
  address   = {Cham},
  pages     = {369--387},
}
```
