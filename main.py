from functools import lru_cache

import ltn
from model.lstm import LSTMModel
import pandas as pd
import numpy as np
import torch
from data.dataset import StandardDataset, CounterfactualDataset, ModelConfig
from torch.utils.data import DataLoader
import logging
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from data.utils import compute_cf_mse
from utils import create_single_new_feat_vec, recommend_act_loader, compute_cf_fairness
from data.utils import compute_average_execution_time
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import random
random.seed(33)
import argparse

import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size of the LSTM model")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of layers in the LSTM model")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for the LSTM model")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--num_epochs_nesy", type=int, default=50, help="Number of epochs for training LTN model")
    parser.add_argument("--train_vanilla", action="store_true", help="Train vanilla LSTM model")
    parser.add_argument("--train_nesy", action="store_true", help="Train LTN model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dataset", type=str, default="bpi12", help="Dataset to use")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for the optimizer")
    parser.add_argument("--abstraction_mode", type=str, default="sequence", help="Abstraction mode for transition system")

    return parser.parse_args()

if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # logger = logging.getLogger(__name__)
    # logging.basicConfig(filename='output.log', level=logging.INFO)

    args = get_args()

    print("INFO: Reading dataset...")
    if args.dataset == "bpi12":
        data = pd.read_pickle('data/bpi12.pkl')
    elif args.dataset == "hospital":
        data = pd.read_csv("data/hospital_log_medium.csv")
    elif args.dataset == "cancer":
        data = pd.read_pickle("data/cancer_screening.pkl")
    elif args.dataset == "hospital_billing_gender":
        data = pd.read_pickle("data/hospital_billing_gender.pkl")
    elif args.dataset == "hospital_billing_age":
        data = pd.read_pickle("data/hospital_billing_age.pkl")
    elif args.dataset == "hospital_billing_age_gender":
        data = pd.read_pickle("data/hospital_billing_age_gender.pkl")
    elif args.dataset == "bpi13":
        data = pd.read_csv("data/bpi13i.csv")
    elif args.dataset == "renting":
        data = pd.read_csv("data/renting_log_high.csv")
        data = data.drop(columns=["activity", "time", "case:protected", "@@index", "case:@@case_index", "case:case"])

    if 'case_id' in data.columns:
        case_id_name = 'case_id'
    elif 'case:concept:name' in data.columns:
        case_id_name = 'case:concept:name'
    else:
        raise ValueError("Could not find case ID column. Available columns: " + str(list(data.columns)))

    if 'activity' in data.columns:
        activity_name = 'activity'
    elif 'concept:name' in data.columns:
        activity_name = 'concept:name'
    else:
        raise ValueError("Could not find activity column. Available columns: " + str(list(data.columns)))
    
    if 'time' in data.columns:
        data.rename(columns={'time': 'time:timestamp'}, inplace=True)

    with open("results.txt", "a") as file:
        file.write(f"Dataset: {args.dataset}, Seed: {args.seed}\n")

    # if not integers, encode case IDs as integers
    data[case_id_name] = data[case_id_name].astype('category').cat.codes + 1
    print("INFO: Dataset read successfully.")
    print("INFO: Preprocessing dataset...")
    if args.dataset == "bpi12" or args.dataset == "bpi12_kappel":
        from data import preprocess_bpi12
        sensitive_attrs = ["gender"]
        (case_ids_train, X_train, X_train_cf, y_train, case_ids_val, X_val, X_val_cf, y_val, case_ids_test, X_test, X_test_cf, y_test, seq_len, feature_names, scalers_training, scalers_val, scalers_test), vocab_sizes, transition_system, activity_mapping = preprocess_bpi12.preprocess_eventlog(data, args.dataset, sensitive_attrs, args.abstraction_mode)
    elif args.dataset in ("hospital_billing_gender", "hospital_billing_age", "hospital_billing_age_gender"):
        from data import preprocess_hospital_billing_log
        sensitive_map = {
            "hospital_billing_gender": ["gender"],
            "hospital_billing_age": ["age"],
            "hospital_billing_age_gender": ["age", "gender"],
        }
        sensitive_attrs = sensitive_map[args.dataset]
        if not np.issubdtype(data[case_id_name].dtype, np.integer):
            data[case_id_name] = data[case_id_name].astype('category').cat.codes + 1
        data[case_id_name] = data[case_id_name].astype(int)
        (case_ids_train, X_train, X_train_cf, y_train, case_ids_val, X_val, X_val_cf, y_val, case_ids_test, X_test, X_test_cf, y_test, seq_len, feature_names, scalers_training, scalers_val, scalers_test), vocab_sizes, transition_system, activity_mapping = preprocess_hospital_billing_log.preprocess_eventlog(data, args.dataset, sensitive_attrs, args.abstraction_mode)
    elif args.dataset == "bpi13":
        from data import preprocess_bpi13
        sensitive_attrs = ["organization country"]
        (case_ids_train, X_train, X_train_cf, y_train, case_ids_val, X_val, X_val_cf, y_val, case_ids_test, X_test, X_test_cf, y_test, seq_len, feature_names, scalers_training, scalers_val, scalers_test), vocab_sizes, transition_system, activity_mapping = preprocess_bpi13.preprocess_eventlog(data, args.dataset, sensitive_attrs, args.abstraction_mode)
    y_train_og = y_train[0]
    y_train_cf = y_train[1]
    y_val_og = y_val[0]
    y_val_cf = y_val[1]
    y_test_og = y_test[0]
    y_test_cf = y_test[1]
    case_ids_train_og = case_ids_train[0]
    case_ids_train_cf = case_ids_train[1]
    case_ids_val_og = case_ids_val[0]
    case_ids_val_cf = case_ids_val[1]
    case_ids_test_og = case_ids_test[0]
    case_ids_test_cf = case_ids_test[1]
    print("INFO: Preprocessing completed.")
    
    print("INFO: Feature names: ", feature_names)

    avg_exec_times = compute_average_execution_time(data, case_col=case_id_name, activity_col=activity_name, timestamp_col="time:timestamp")
    print("INFO: Average execution times computed.")

    config = ModelConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        num_epochs = args.num_epochs,
        dataset = args.dataset,
        sequence_length = seq_len,
        seed = args.seed
    )

    train_dataset_og = StandardDataset(X_train, y_train_og, case_ids_train_og)
    val_dataset_og = StandardDataset(X_val, y_val_og, case_ids_val_og)
    test_dataset_og = StandardDataset(X_test, y_test_og, case_ids_test_og)

    train_dataset_cf = CounterfactualDataset(X_train_cf, y_train_og, case_ids_train_cf)
    val_dataset_cf = CounterfactualDataset(X_val_cf, y_val_og, case_ids_val_cf)
    test_dataset_cf = CounterfactualDataset(X_test_cf, y_test_og, case_ids_test_cf)

    train_loader = DataLoader(train_dataset_og, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset_og, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset_og, batch_size=32, shuffle=False)

    train_loader_cf = DataLoader(train_dataset_cf, batch_size=32, shuffle=True)
    val_loader_cf = DataLoader(val_dataset_cf, batch_size=32, shuffle=False)
    test_loader_cf = DataLoader(test_dataset_cf, batch_size=32, shuffle=False)

    if args.train_vanilla:
        lstm = LSTMModel(vocab_sizes, config, 1, feature_names).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(lstm.parameters(), lr=args.learning_rate)
        print("INFO: Starting training vanilla baseline...")
        for epoch in range(0, args.num_epochs):
            lstm.train()
            total_loss = 0.0
            total_validation_loss = 0.0
            for batch_idx, (x, y, case_id) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = lstm(x)
                loss = criterion(outputs.squeeze(), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            lstm.eval()
            with torch.no_grad():
                for batch_idx, (x, y, case_id) in enumerate(val_loader):
                    x, y = x.to(device), y.to(device)
                    outputs = lstm(x)
                    val_loss = criterion(outputs.squeeze(), y)
                    total_validation_loss += val_loss.item()
            avg_val_loss = total_validation_loss / len(val_loader)
            print(f"INFO: Epoch [{epoch+1}/{args.num_epochs}], Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            if epoch == 0:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(lstm.state_dict(), 'best_model.pt')
            else:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(lstm.state_dict(), 'best_model.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= 15:
                        print(f"INFO: Early stopping triggered at epoch {epoch+1}")
                        lstm.load_state_dict(torch.load('best_model.pt'))
                        break
        print("INFO: Training of vanilla baseline completed!")
        print("INFO: Starting evaluation on test set...")
        lstm.load_state_dict(torch.load('best_model.pt'))
        lstm.eval()
        maes = []
        rmses = []
        predictions = []
        predictions_og = []
        predictions_cf = []
        gt_labels = []
        case_ids_list = []
        with torch.no_grad():
            for batch_idx, (x, x_cf, y, case_id) in enumerate(test_loader_cf):
                x, x_cf, y = x.to(device), x_cf.to(device), y.to(device)
                x_cf = x_cf.squeeze(1)
                y_pred = lstm(x)
                y_pred = scalers_test['remaining_time'].inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))
                y_pred = np.maximum(y_pred, 0)
                predictions_og.extend(y_pred.flatten().tolist())
                y_true = scalers_test['remaining_time'].inverse_transform(y.cpu().numpy().reshape(-1, 1))
                gt_labels.extend(y_true.flatten().tolist())
                y_cf_pred = lstm(x_cf)
                y_cf_pred = scalers_test['remaining_time'].inverse_transform(y_cf_pred.cpu().numpy().reshape(-1, 1))
                y_cf_pred = np.maximum(y_cf_pred, 0)
                predictions_cf.extend(y_cf_pred.flatten().tolist())
                case_ids_list.extend(case_id.cpu().numpy().tolist())
        
        # Write results to CSV
        df_results = pd.DataFrame({
            'case_id': case_ids_list,
            'y_true': gt_labels,
            'y_pred': predictions_og,
            'y_cf_pred': predictions_cf
        })
        df_results.to_csv(f'run_seed_{args.seed}/{args.dataset}_LSTM_predictions.csv', index=False)
        print(f"INFO: Baseline Test MAE: {mean_absolute_error(gt_labels, predictions_og):.4f}, RMSE: {root_mean_squared_error(gt_labels, predictions_og):.4f}, CF MSE: {compute_cf_mse(predictions_og, predictions_cf):.4f}")
        print("INFO: Generating recommendations and evaluating counterfactual fairness...")
        act_kpi_og_final_results, act_kpi_cf_final_results = recommend_act_loader(test_loader_cf, transition_system, lstm, activity_mapping, seq_len, feature_names, avg_exec_times, scalers_test, device, args.dataset, "LSTM", args.seed, save_to_file=True)
        cf_fairness = compute_cf_fairness(act_kpi_og_final_results, act_kpi_cf_final_results)
        print(f"INFO: Baseline Counterfactual Fairness: {cf_fairness:.4f}")
        with open("results.txt", "a") as file:
            file.write(f"Baseline Test MAE: {mean_absolute_error(gt_labels, predictions_og):.4f}, RMSE: {root_mean_squared_error(gt_labels, predictions_og):.4f}, CF MSE: {compute_cf_mse(predictions_og, predictions_cf):.4f}, CF Fairness: {cf_fairness:.4f}\n")
    
    if args.train_nesy:
        # LTN Knowledge Base
        Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
        Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        And = ltn.Connective(ltn.fuzzy_ops.AndProd())
        Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
        Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
        alpha = 0.05
        Eq = ltn.Predicate(func = lambda u, v: torch.exp(-alpha * torch.sqrt(torch.sum(torch.square(u - v), dim=1) + 1e-10 )))
        Equal = ltn.Predicate(func = lambda u, v: (u == v).float())
        reverse_activity_mapping = {v: k for k, v in activity_mapping.items()}

        @lru_cache(maxsize=200_000)
        def cached_next_acts(prefix_tuple):
            """
            Cache transition_system.get_val(prefix) results.
            prefix_tuple must be hashable (tuple of activity labels/strings).
            """
            next_acts = transition_system.get_val(list(prefix_tuple))
            if isinstance(next_acts, list):
                # tuple for caching + immutability
                return tuple(next_acts)
            return tuple()

        def recommend_act(x):
            """
            Optimized recommend_act:
            - Pre-computed reverse mapping (no O(n) scan per call)
            - Single batched model forward pass for ALL (sample × candidate) pairs
            - Vectorized numpy inverse-transform
            - torch.no_grad() scoped tightly around inference only
            """
            n = len(x)
            results = [None] * n  # pre-allocate

            # ── 1. Collect all (sample_idx, insert_idx, act_int, feat_vec) tuples ──
            all_feat_vecs = []
            meta = []  # (sample_idx, act_int)

            for i in range(n):
                seq = x[i][:seq_len]
                nonzero_mask = seq != 0
                zero_indices = (~nonzero_mask).nonzero(as_tuple=True)[0]

                if len(zero_indices) == 0:
                    results[i] = torch.tensor(0, device=device)
                    continue

                insert_idx = zero_indices[0]
                prefix = [activity_mapping[int(a)] for a in seq[nonzero_mask].tolist()]
                next_acts = transition_system.get_val(prefix)

                if not isinstance(next_acts, list) or len(next_acts) == 0:
                    results[i] = torch.tensor(0, device=device)
                    continue

                for act in next_acts:
                    act_int = reverse_activity_mapping[act]
                    feat_vec = create_single_new_feat_vec(
                        x[i], act_int, insert_idx, seq_len,
                        feature_names, avg_exec_times.get(act, 0), scalers_training
                    )
                    all_feat_vecs.append(feat_vec)
                    meta.append((i, act_int))

            # ── 2. Single batched forward pass ──────────────────────────────────────
            if all_feat_vecs:
                with torch.no_grad():
                    batch_tensor = torch.stack(all_feat_vecs).to(device)
                    preds = model(batch_tensor).squeeze(-1)

                preds_np = scalers_training['remaining_time'].inverse_transform(
                    preds.cpu().numpy().reshape(-1, 1)
                ).flatten()
                preds_np = np.maximum(preds_np, 0)

                # ── 3. Pick best (lowest KPI) action per sample ────────────────────
                best = {}  # sample_idx -> (act_int, kpi)
                for flat_idx, (sample_idx, act_int) in enumerate(meta):
                    kpi = preds_np[flat_idx]
                    if sample_idx not in best or kpi < best[sample_idx][1]:
                        best[sample_idx] = (act_int, kpi)

                for sample_idx, (act_int, _) in best.items():
                    results[sample_idx] = torch.tensor(act_int, device=device)

            # ── 4. Fill any remaining None (shouldn't happen, safety net) ──────────
            for i in range(n):
                if results[i] is None:
                    results[i] = torch.tensor(0, device=device)

            return torch.stack(results)
        def recommend_act_soft(x, tau=1.0):
            batch_size = len(x)
            all_feat_vecs = []
            meta = []  # (sample_idx, num_candidates) to reconstruct results after forward pass
            fallback = {}  # sample_idx -> zero tensor for samples with no valid candidates

            # ── 1. Build ALL feature vectors across the entire batch ────────────────
            for i in range(batch_size):
                seq = x[i][:seq_len]
                nonzero_mask = seq != 0
                zero_indices = (~nonzero_mask).nonzero(as_tuple=True)[0]

                if len(zero_indices) == 0:
                    fallback[i] = torch.zeros(1, device=device)
                    continue

                insert_idx = zero_indices[0]
                prefix = [activity_mapping[int(a)] for a in seq[nonzero_mask].tolist()]
                next_acts = transition_system.get_val(prefix)

                if not isinstance(next_acts, list) or len(next_acts) == 0:
                    fallback[i] = torch.zeros(1, device=device)
                    continue

                candidate_vecs = []
                for act in next_acts:
                    act_int = reverse_activity_mapping[act]
                    feat_vec = create_single_new_feat_vec(
                        x[i], act_int, insert_idx, seq_len,
                        feature_names, avg_exec_times.get(act, 0), scalers_training
                    )
                    candidate_vecs.append(feat_vec)

                all_feat_vecs.extend(candidate_vecs)
                meta.append((i, len(candidate_vecs)))  # remember how many candidates this sample has

            # ── 2. Single forward pass for ALL candidates across ALL samples ─────────
            if all_feat_vecs:
                batch_tensor = torch.stack(all_feat_vecs).to(device)  # [total_candidates, ...]
                all_kpis = model(batch_tensor).squeeze(-1)             # [total_candidates]
            
            # ── 3. Reconstruct per-sample soft KPI using meta offsets ────────────────
            results = {}
            offset = 0
            for sample_idx, num_candidates in meta:
                kpis = all_kpis[offset: offset + num_candidates]  # slice this sample's KPIs
                offset += num_candidates

                if num_candidates == 1:
                    results[sample_idx] = kpis  # shape [1], already in graph
                else:
                    weights = torch.softmax(-kpis / tau, dim=0)
                    results[sample_idx] = (weights * kpis).sum().unsqueeze(0)

            # ── 4. Assemble final output in original batch order ─────────────────────
            output = [
                results[i] if i in results else fallback[i]
                for i in range(batch_size)
            ]

            return torch.stack(output)  # [batch_size, 1]

        ReccAct = ltn.Function(func = lambda u: recommend_act(u))
        ReccAct_soft = ltn.Function(func=lambda u: recommend_act_soft(u))
        SatAgg = ltn.fuzzy_ops.SatAgg()
        model = LSTMModel(vocab_sizes, config, 1, feature_names).to(device)
        f = ltn.Function(model)
        params = list(f.parameters())
        optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=1e-5)
        def create_recommended_samples(u):
            batch_size = len(u)
            all_feat_vecs = []
            meta = []  # (sample_idx, num_candidates) to reconstruct results after forward pass
            fallback = {}  # sample_idx -> zero tensor for samples with no valid candidates

            # ── 1. Build ALL feature vectors across the entire batch ────────────────
            for i in range(batch_size):
                seq = u[i][:seq_len]
                nonzero_mask = seq != 0
                zero_indices = (~nonzero_mask).nonzero(as_tuple=True)[0]

                if len(zero_indices) == 0:
                    fallback[i] = torch.zeros(1, device=device)
                    continue

                insert_idx = zero_indices[0]
                prefix = [activity_mapping[int(a)] for a in seq[nonzero_mask].tolist()]
                next_acts = transition_system.get_val(prefix)

                if not isinstance(next_acts, list) or len(next_acts) == 0:
                    fallback[i] = torch.zeros(1, device=device)
                    continue

                candidate_vecs = []
                for act in next_acts:
                    act_int = reverse_activity_mapping[act]
                    feat_vec = create_single_new_feat_vec(
                        u[i], act_int, insert_idx, seq_len,
                        feature_names, avg_exec_times.get(act, 0), scalers_training
                    )
                    candidate_vecs.append(feat_vec)
                all_feat_vecs.extend(candidate_vecs)
            if all_feat_vecs:
                batch_tensor = torch.stack(all_feat_vecs).to(device)
                return batch_tensor
            else:
                return torch.tensor([], device=device)
        def compute_regr_fair_sat_level(loader):
            mean_sat_regr = 0
            mean_sat_fair = 0
            mean_sat_fair_recc = 0
            mean_sat_fair_kpis = 0
            with torch.no_grad():
                for x_data, x_cf_data, y_data, _ in loader:
                    x_data.to(device), x_cf_data.to(device), y_data.to(device)
                    x_data_recommended = create_recommended_samples(x_data)
                    x_cf_data_recommended = create_recommended_samples(x_cf_data.view(-1, x_cf_data.shape[-1]))
                    num_counterfactual = x_cf_data.shape[1]
                    x_cf_data = x_cf_data.view(-1, x_cf_data.shape[-1])
                    x_data_repeated = x_data.repeat_interleave(num_counterfactual, dim=0)
                    x = ltn.Variable("x", x_data)
                    x_cf = ltn.Variable("x_cf", x_cf_data)
                    x_repeated = ltn.Variable("x_repeated", x_data_repeated)
                    x_recommended = ltn.Variable("x_recommended", x_data_recommended)
                    x_cf_recommended = ltn.Variable("x_cf_recommended", x_cf_data_recommended)
                    y = ltn.Variable("y", y_data)
                    mean_sat_regr += Forall(ltn.diag(x, y), Eq(f(x), y)).value
                    mean_sat_fair += Forall(ltn.diag(x_repeated, x_cf), Eq(f(x_repeated), f(x_cf))).value
                    # mean_sat_fair_recc += Forall(ltn.diag(x_repeated, x_cf), Eq(ReccAct_soft(x_repeated), ReccAct_soft(x_cf))).value
                    mean_sat_fair_kpis += Forall(ltn.diag(x_recommended, x_cf_recommended), Eq(f(x_recommended), f(x_cf_recommended))).value
            mean_sat_regr /= len(loader)
            mean_sat_fair /= len(loader)
            mean_sat_fair_recc /= len(loader)
            mean_sat_fair_kpis /= len(loader)
            return mean_sat_regr, mean_sat_fair, mean_sat_fair_recc, mean_sat_fair_kpis
        def compute_regr_fair_sat_level_(loader):
            mean_sat_regr = 0
            mean_sat_fair = 0
            with torch.no_grad():
                for x_data, x_cf_data, y_data, _ in loader:
                    x_data.to(device), x_cf_data.to(device), y_data.to(device)
                    num_counterfactual = x_cf_data.shape[1]
                    x_cf_data = x_cf_data.view(-1, x_cf_data.shape[-1])
                    x_data_repeated = x_data.repeat_interleave(num_counterfactual, dim=0)
                    x = ltn.Variable("x", x_data)
                    x_cf = ltn.Variable("x_cf", x_cf_data)
                    x_repeated = ltn.Variable("x_repeated", x_data_repeated)
                    y = ltn.Variable("y", y_data)
                    mean_sat_regr += Forall(ltn.diag(x, y), Eq(f(x), y)).value
                    mean_sat_fair += Forall(ltn.diag(x_repeated, x_cf), Eq(f(x_repeated), f(x_cf))).value
            mean_sat_regr /= len(loader)
            mean_sat_fair /= len(loader)
            return mean_sat_regr, mean_sat_fair
        pareto_archive_val = []
        model_checkpoints = []
        val_history = []
        patience = 10
        last_val_improvement = 0
        model = LSTMModel(vocab_sizes, config, 1, feature_names).to(device)
        f = ltn.Function(model)
        params = list(f.parameters())
        optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=1e-5)
        print("INFO: Training with regression and prediction fairness axioms using Pareto front monitoring (Single Stage)...")
        for epoch in range(1, args.num_epochs_nesy + 1):
            model.train()
            total_loss = 0.0
            for batch_idx, (x_data, x_cf_data, y_data, case_id) in enumerate(train_loader_cf):
                x_data, x_cf_data, y_data = x_data.to(device), x_cf_data.to(device), y_data.to(device)
                optimizer.zero_grad()
                num_counterfactual = x_cf_data.shape[1]
                x_cf_data = x_cf_data.view(-1, x_cf_data.shape[-1])
                x_data_repeated = x_data.repeat_interleave(num_counterfactual, dim=0)
                x = ltn.Variable("x", x_data)
                x_cf = ltn.Variable("x_cf", x_cf_data)
                x_repeated = ltn.Variable("x_repeated", x_data_repeated)
                y = ltn.Variable("y", y_data)
                sat_regression = Forall(ltn.diag(x, y), Eq(f(x), y)).value
                sat_fairness = Forall(ltn.diag(x_repeated, x_cf), Eq(f(x_repeated), f(x_cf))).value
                sat_agg = SatAgg(*[sat_regression, sat_fairness])
                loss = 1.0 - sat_agg
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader_cf)
            if epoch % 1 == 0:
                # sat_regr_train, sat_fair_train, sat_fair_recc_train = compute_regr_fair_sat_level(train_loader_cf)
                sat_regr_val, sat_fair_val = compute_regr_fair_sat_level_(val_loader_cf)
                # sat_regr_test, sat_fair_test, sat_fair_recc_test = compute_regr_fair_sat_level(test_loader_cf)
                # print(f"INFO: Epoch [{epoch}/{args.num_epochs_nesy}], Loss: {avg_loss:.4f}, Train Regr Sat: {sat_regr_train:.4f}, Val Regr Sat: {sat_regr_val:.4f}, Test Regr Sat: {sat_regr_test:.4f}, Train Fair Sat: {sat_fair_train:.4f}, Val Fair Sat: {sat_fair_val:.4f}, Test Fair Sat: {sat_fair_test:.4f}")
                print(f"INFO: Epoch [{epoch}/{args.num_epochs_nesy}]")
                current_val = [sat_regr_val.item(), sat_fair_val.item()]
                val_history.append(current_val)
                dominated = False
                for i, p in enumerate(pareto_archive_val):
                    if all(current_val[j] >= p[j] for j in range(2)):
                        pareto_archive_val.pop(i)
                    elif all(p[j] >= current_val[j] for j in range(2)):
                        dominated = True
                        break
                if not dominated:
                    pareto_archive_val.append(current_val)
                    model_path = f'single_fair_prediction_best_pareto_model_e{epoch}_sats{current_val}.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_sats': current_val,
                        'pareto_archive': pareto_archive_val.copy()
                    }, model_path)
                    model_checkpoints.append(model_path)
                    last_val_improvement = epoch
                    print(f"INFO: New Pareto front found at epoch {epoch}, saved model to {model_path}")
                if epoch - last_val_improvement >= patience:
                    print(f"INFO: Early stopping triggered at epoch {epoch+1}. Val Pareto stable for {patience}")
                    break
        pareto_models = []
        for model_path in model_checkpoints:
            checkpoint = torch.load(model_path)
            sats = checkpoint['val_sats']
            pareto_models.append((sats, model_path))
        
        # Best balanced model (highest minimum of both satisfactions)
        best_idx = np.argmax([min(sats) for sats, _ in pareto_models])
        best_model_path = pareto_models[best_idx][1]
        print(f"INFO: Loading best balanced Pareto model from {best_model_path} for final evaluation.")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("INFO: Starting evaluation on test set...")
        model.eval()
        predictions = []
        predictions_cf = []
        gt_labels = []
        with torch.no_grad():
            for batch_idx, (x, x_cf, y, case_id) in enumerate(test_loader_cf):
                x, x_cf, y = x.to(device), x_cf.to(device), y.to(device)
                x_cf = x_cf.squeeze(1)
                y_pred = model(x)
                y_pred = scalers_test['remaining_time'].inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))
                y_pred = np.maximum(y_pred, 0)
                predictions.extend(y_pred.flatten().tolist())
                y_true = scalers_test['remaining_time'].inverse_transform(y.cpu().numpy().reshape(-1, 1))
                gt_labels.extend(y_true.flatten().tolist())
                y_cf_pred = model(x_cf)
                y_cf_pred = scalers_test['remaining_time'].inverse_transform(y_cf_pred.cpu().numpy().reshape(-1, 1))
                y_cf_pred = np.maximum(y_cf_pred, 0)
                predictions_cf.extend(y_cf_pred.flatten().tolist())        
        df_results = pd.DataFrame({
            'case_id': case_ids_test_cf,
            'y_true': gt_labels,
            'y_pred': predictions,
            'y_cf_pred': predictions_cf
        })
        df_results.to_csv(f'run_seed_{args.seed}/{args.dataset}_LTN_Semi_Fairness_predictions.csv', index=False)
        print("INFO: Evaluation completed!")
        print(f"INFO: LTN + Fairness (Single Stage) Test MAE: {mean_absolute_error(gt_labels, predictions):.4f}, RMSE: {root_mean_squared_error(gt_labels, predictions):.4f}, CF MSE: {compute_cf_mse(predictions, predictions_cf):.4f}")
        print("INFO: Generating recommendations and evaluating counterfactual fairness...")
        act_kpi_og_final_results, act_kpi_cf_final_results = recommend_act_loader(test_loader_cf, transition_system, model, activity_mapping, seq_len, feature_names, avg_exec_times, scalers_test, device, args.dataset, "LTN_Single_Stage_Fairness", args.seed, save_to_file=True)
        cf_fairness = compute_cf_fairness(act_kpi_og_final_results, act_kpi_cf_final_results)
        print(f"INFO: LTN + Fairness (Single Stage) Counterfactual Fairness: {cf_fairness:.4f}")
        with open("results.txt", "a") as file:
            file.write(f"LTN + Fairness (Single Stage) Test MAE: {mean_absolute_error(gt_labels, predictions):.4f}, RMSE: {root_mean_squared_error(gt_labels, predictions):.4f}, CF MSE: {compute_cf_mse(predictions, predictions_cf):.4f}, CF Fairness: {cf_fairness:.4f}\n")
        
        val_history = []
        patience = 10
        last_val_improvement = 0
        model = LSTMModel(vocab_sizes, config, 1, feature_names).to(device)
        f = ltn.Function(model)
        params = list(f.parameters())
        optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=1e-5)
        print("INFO: Starting training LTN model (Two-Stage)...")
        print("INFO: Training with regression axioms only...")
        for epoch in range(1, args.num_epochs_nesy + 1):
            model.train()
            total_loss = 0.0
            total_validation_loss = 0.0
            for batch_idx, (x_data, y_data, case_id) in enumerate(train_loader):
                x_data, y_data = x_data.to(device), y_data.to(device)
                optimizer.zero_grad()
                x = ltn.Variable("x", x_data)
                y = ltn.Variable("y", y_data)
                sat_agg = Forall(ltn.diag(x, y), Eq(f(x), y)).value
                loss = 1.0 - sat_agg
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            model.eval()
            with torch.no_grad():
                for x_data, y_data, _ in val_loader:
                    x_data.to(device), y_data.to(device)
                    x = ltn.Variable("x", x_data)
                    y = ltn.Variable("y", y_data)
                    val_loss = 1.0 - Forall(ltn.diag(x, y), Eq(f(x), y)).value
                    total_validation_loss += val_loss.item()
            avg_val_loss = total_validation_loss / len(val_loader)
            # if epoch % 1 == 0:
                # sat_regr_train, sat_fair_train, sat_fair_recc_train = compute_regr_fair_sat_level(train_loader_cf)
                # sat_regr_val, sat_fair_val, sat_fair_recc_val = compute_regr_fair_sat_level(val_loader_cf)
                # sat_regr_test, sat_fair_test, sat_fair_recc_test = compute_regr_fair_sat_level(test_loader_cf)
                # print(f"INFO: Epoch [{epoch}/{args.num_epochs_nesy}], Loss: {avg_loss:.4f}, Train Regr Sat: {sat_regr_train:.4f}, Val Regr Sat: {sat_regr_val:.4f}, Test Regr Sat: {sat_regr_test:.4f}, Train Fair Sat: {sat_fair_train:.4f}, Val Fair Sat: {sat_fair_val:.4f}, Test Fair Sat: {sat_fair_test:.4f}")
            print(f"INFO: Epoch [{epoch}/{args.num_epochs_nesy}]")
            if epoch == 1:
                best_val_loss_nesy = avg_val_loss
                patience_counter_nesy = 0
                torch.save(model.state_dict(), 'best_model_nesy_regr.pt')
            else:
                if avg_val_loss < best_val_loss_nesy:
                    best_val_loss_nesy = avg_val_loss
                    patience_counter_nesy = 0
                    torch.save(model.state_dict(), 'best_model_nesy_regr.pt')
                else:
                    patience_counter_nesy += 1
                    if patience_counter_nesy >= 10:
                        print(f"INFO: Early stopping triggered at epoch {epoch}")
                        model.load_state_dict(torch.load('best_model_nesy_regr.pt'))
                        break
        model.load_state_dict(torch.load('best_model_nesy_regr.pt'))
        print("INFO: Starting evaluation on test set...")
        model.eval()
        predictions = []
        predictions_cf = []
        gt_labels = []
        with torch.no_grad():
            for batch_idx, (x, x_cf, y, case_id) in enumerate(test_loader_cf):
                x, x_cf, y = x.to(device), x_cf.to(device), y.to(device)
                x_cf = x_cf.squeeze(1)
                y_pred = model(x)
                y_pred = scalers_test['remaining_time'].inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))
                y_pred = np.maximum(y_pred, 0)
                predictions.extend(y_pred.flatten().tolist())
                y_true = scalers_test['remaining_time'].inverse_transform(y.cpu().numpy().reshape(-1, 1))
                gt_labels.extend(y_true.flatten().tolist())
                y_cf_pred = model(x_cf)
                y_cf_pred = scalers_test['remaining_time'].inverse_transform(y_cf_pred.cpu().numpy().reshape(-1, 1))
                y_cf_pred = np.maximum(y_cf_pred, 0)
                predictions_cf.extend(y_cf_pred.flatten().tolist())
        df_results = pd.DataFrame({
            'case_id': case_ids_test_cf,
            'y_true': gt_labels,
            'y_pred': predictions,
            'y_cf_pred': predictions_cf
        })
        df_results.to_csv(f'run_seed_{args.seed}/{args.dataset}_LTN_predictions.csv', index=False)
        
        print("INFO: Evaluation completed!")
        print(f"INFO: LTN Test MAE: {mean_absolute_error(gt_labels, predictions):.4f}, RMSE: {root_mean_squared_error(gt_labels, predictions):.4f}, CF MSE: {compute_cf_mse(predictions, predictions_cf):.4f}")
        print("INFO: Generating recommendations and evaluating counterfactual fairness...")
        act_kpi_og_final_results, act_kpi_cf_final_results = recommend_act_loader(test_loader_cf, transition_system, model, activity_mapping, seq_len, feature_names, avg_exec_times, scalers_test, device, args.dataset, "LTN", args.seed, save_to_file=True)
        cf_fairness = compute_cf_fairness(act_kpi_og_final_results, act_kpi_cf_final_results)
        print(f"INFO: LTN Counterfactual Fairness: {cf_fairness:.4f}")
        with open("results.txt", "a") as file:
            file.write(f"LTN Test MAE: {mean_absolute_error(gt_labels, predictions):.4f}, RMSE: {root_mean_squared_error(gt_labels, predictions):.4f}, CF MSE: {compute_cf_mse(predictions, predictions_cf):.4f}, CF Fairness: {cf_fairness:.4f}\n")
        
        pareto_archive_val = []
        model_checkpoints = []
        val_history = []
        patience = 10
        last_val_improvement = 0

        print("INFO: Training with regression and prediction fairness axioms using Pareto front monitoring...")
        for epoch in range(1, args.num_epochs_nesy + 1):
            model.train()
            total_loss = 0.0
            for batch_idx, (x_data, x_cf_data, y_data, case_id) in enumerate(train_loader_cf):
                x_data, x_cf_data, y_data = x_data.to(device), x_cf_data.to(device), y_data.to(device)
                optimizer.zero_grad()
                num_counterfactual = x_cf_data.shape[1]
                x_cf_data = x_cf_data.view(-1, x_cf_data.shape[-1])
                x_data_repeated = x_data.repeat_interleave(num_counterfactual, dim=0)
                x = ltn.Variable("x", x_data)
                x_cf = ltn.Variable("x_cf", x_cf_data)
                x_repeated = ltn.Variable("x_repeated", x_data_repeated)
                y = ltn.Variable("y", y_data)
                # sat_regression = Forall(ltn.diag(x, y), Eq(f(x), y)).value
                sat_fairness = Forall(ltn.diag(x_repeated, x_cf), Eq(f(x_repeated), f(x_cf))).value
                sat_agg = SatAgg(*[sat_fairness])
                loss = 1.0 - sat_agg
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader_cf)
            if epoch % 1 == 0:
                # sat_regr_train, sat_fair_train, sat_fair_recc_train = compute_regr_fair_sat_level(train_loader_cf)
                sat_regr_val, sat_fair_val = compute_regr_fair_sat_level_(val_loader_cf)
                # sat_regr_test, sat_fair_test, sat_fair_recc_test = compute_regr_fair_sat_level(test_loader_cf)
                # print(f"INFO: Epoch [{epoch}/{args.num_epochs_nesy}], Loss: {avg_loss:.4f}, Train Regr Sat: {sat_regr_train:.4f}, Val Regr Sat: {sat_regr_val:.4f}, Test Regr Sat: {sat_regr_test:.4f}, Train Fair Sat: {sat_fair_train:.4f}, Val Fair Sat: {sat_fair_val:.4f}, Test Fair Sat: {sat_fair_test:.4f}")
                print(f"INFO: Epoch [{epoch}/{args.num_epochs_nesy}]")
                current_val = [sat_regr_val.item(), sat_fair_val.item()]
                val_history.append(current_val)
                dominated = False
                for i, p in enumerate(pareto_archive_val):
                    if all(current_val[j] >= p[j] for j in range(2)):
                        pareto_archive_val.pop(i)
                    elif all(p[j] >= current_val[j] for j in range(2)):
                        dominated = True
                        break
                if not dominated:
                    pareto_archive_val.append(current_val)
                    model_path = f'fair_prediction_best_pareto_model_e{epoch}_sats{current_val}.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_sats': current_val,
                        'pareto_archive': pareto_archive_val.copy()
                    }, model_path)
                    model_checkpoints.append(model_path)
                    last_val_improvement = epoch
                    print(f"INFO: New Pareto front found at epoch {epoch}, saved model to {model_path}")
                if epoch - last_val_improvement >= patience:
                    print(f"INFO: Early stopping triggered at epoch {epoch+1}. Val Pareto stable for {patience}")
                    break
        pareto_models = []
        for model_path in model_checkpoints:
            checkpoint = torch.load(model_path)
            sats = checkpoint['val_sats']
            pareto_models.append((sats, model_path))
        
        # Best balanced model (highest minimum of both satisfactions)
        best_idx = np.argmax([min(sats) for sats, _ in pareto_models])
        best_model_path = pareto_models[best_idx][1]
        print(f"INFO: Loading best balanced Pareto model from {best_model_path} for final evaluation.")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("INFO: Starting evaluation on test set...")
        model.eval()
        predictions = []
        predictions_cf = []
        gt_labels = []
        with torch.no_grad():
            for batch_idx, (x, x_cf, y, case_id) in enumerate(test_loader_cf):
                x, x_cf, y = x.to(device), x_cf.to(device), y.to(device)
                x_cf = x_cf.squeeze(1)
                y_pred = model(x)
                y_pred = scalers_test['remaining_time'].inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))
                y_pred = np.maximum(y_pred, 0)
                predictions.extend(y_pred.flatten().tolist())
                y_true = scalers_test['remaining_time'].inverse_transform(y.cpu().numpy().reshape(-1, 1))
                gt_labels.extend(y_true.flatten().tolist())
                y_cf_pred = model(x_cf)
                y_cf_pred = scalers_test['remaining_time'].inverse_transform(y_cf_pred.cpu().numpy().reshape(-1, 1))
                y_cf_pred = np.maximum(y_cf_pred, 0)
                predictions_cf.extend(y_cf_pred.flatten().tolist())        
        df_results = pd.DataFrame({
            'case_id': case_ids_test_cf,
            'y_true': gt_labels,
            'y_pred': predictions,
            'y_cf_pred': predictions_cf
        })
        df_results.to_csv(f'run_seed_{args.seed}/{args.dataset}_LTN_Two-Stage_Fairness_predictions.csv', index=False)
        print("INFO: Evaluation completed!")
        print(f"INFO: LTN + Fairness (Two-Stage) Test MAE: {mean_absolute_error(gt_labels, predictions):.4f}, RMSE: {root_mean_squared_error(gt_labels, predictions):.4f}, CF MSE: {compute_cf_mse(predictions, predictions_cf):.4f}")
        print("INFO: Generating recommendations and evaluating counterfactual fairness...")
        act_kpi_og_final_results, act_kpi_cf_final_results = recommend_act_loader(test_loader_cf, transition_system, model, activity_mapping, seq_len, feature_names, avg_exec_times, scalers_test, device, args.dataset, "LTN_Semi_Fairness", args.seed, save_to_file=True)
        cf_fairness = compute_cf_fairness(act_kpi_og_final_results, act_kpi_cf_final_results)
        print(f"INFO: LTN + Fairness (Two-Stage) Counterfactual Fairness: {cf_fairness:.4f}")
        with open("results.txt", "a") as file:
            file.write(f"LTN + Fairness (Two-Stage) Test MAE: {mean_absolute_error(gt_labels, predictions):.4f}, RMSE: {root_mean_squared_error(gt_labels, predictions):.4f}, CF MSE: {compute_cf_mse(predictions, predictions_cf):.4f}, CF Fairness: {cf_fairness:.4f}\n")
        val_data = np.array(val_history)
        pareto_data = np.array(pareto_archive_val)
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2)
        plt.figure(figsize=(10, 8), dpi=300)
        df_val = pd.DataFrame(val_data, columns=['Regression Satisfaction', 'Fairness Satisfaction'])
        sns.scatterplot(data=df_val, x='Regression Satisfaction', y='Fairness Satisfaction', s=80, color='lightblue', alpha=0.7, edgecolor='navy', linewidth=0.8, label='Validation Epochs')
        df_pareto = pd.DataFrame(pareto_data, columns=['Regression Satisfaction', 'Fairness Satisfaction'])
        sort_idx = np.argsort(df_pareto['Regression Satisfaction'])[::-1]
        pareto_sorted = df_pareto.iloc[sort_idx].reset_index(drop=True)
        sns.lineplot(data=pareto_sorted, x='Regression Satisfaction', y='Fairness Satisfaction',
                color='red', linewidth=5, marker='o', markersize=12,
                markeredgecolor='darkred', markeredgewidth=2,
                label='Validation Pareto front')
        plt.xlabel('Regression Axioms Satisfaction', fontsize=14, fontweight='bold')
        plt.ylabel('Fairness Axioms Satisfaction', fontsize=14, fontweight='bold')
        plt.title('LTN Training: Pareto Front Monitoring (Validation)', 
                fontsize=16, fontweight='bold', pad=20)

        # Legend styling
        plt.legend(loc='lower right', fontsize=11, frameon=True, fancybox=True, shadow=True)

        # Set axis limits and grid
        if len(val_data) > 0 and len(pareto_data) > 0:
            all_x = np.concatenate([val_data[:,0], pareto_data[:,0]])
            all_y = np.concatenate([val_data[:,1], pareto_data[:,1]])
        else:
            all_x = val_data[:,0] if len(val_data) > 0 else pareto_data[:,0]
            all_y = val_data[:,1] if len(val_data) > 0 else pareto_data[:,1]

        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()
        padding = 0.05  # 5% padding

        plt.xlim(x_min - (x_max - x_min)*padding, x_max + (x_max - x_min)*padding)
        plt.ylim(y_min - (y_max - y_min)*padding, y_max + (y_max - y_min)*padding)
        
        sns.despine()  # Clean border

        # Tight layout for perfect spacing
        plt.tight_layout()

        # Save high-quality PDF (vector graphics, publication ready)
        plt.savefig('ltn_pareto_front.pdf', format='pdf', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none', backend='pdf')