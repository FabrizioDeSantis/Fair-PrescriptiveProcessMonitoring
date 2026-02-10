import pickle
from pathlib import Path

import pandas as pd
import torch


def load_running_log(running_idx_dir="experiments/bpi12_kappel/running_idx", data_dir="data"):
    """
    Load prefixes_test and test_ids from running_idx, load bpi12 from data,
    then build a dataframe of prefix rows: for each test trace, the first 'a' rows
    (a = number of non-zero elements in the first entry of that prefix). Returns
    the concatenated running log as a DataFrame.
    """
    base = Path(running_idx_dir)
    with open(base / "prefixes_test.pkl", "rb") as f:
        prefixes_test = pickle.load(f)
    with open(base / "test_ids.pkl", "rb") as f:
        test_ids = pickle.load(f)
    with open(Path(data_dir) / "bpi12.pkl", "rb") as f:
        bpi12 = pickle.load(f)

    case_id_col = "case_id" if "case_id" in bpi12.columns else "case:concept:name"
    case_ids = bpi12[case_id_col].astype(int) if bpi12[case_id_col].dtype == object else bpi12[case_id_col]

    all_trace_rows = []
    for n in range(len(test_ids)):
        trace_id = test_ids[n]
        first_entry = prefixes_test[n][0]
        a = sum(1 for x in first_entry if x != 0)
        trace_rows = bpi12[case_ids == trace_id].head(a)
        all_trace_rows.append(trace_rows)

    return pd.concat(all_trace_rows, ignore_index=True)

def add_start_timestamp(df):
    """
    Adds a column 'start:timestamp' to the dataframe.
    For the first event in each trace (with smallest 'time:timestamp' for each 'case_id'),
    the value is 0. For the rest, it is the value of the previous event's 'time:timestamp' within the trace.
    """
    df_sorted = df.sort_values(['case_id', 'time:timestamp'])
    df_sorted['start:timestamp'] = df_sorted.groupby('case_id')['time:timestamp'].shift(1)
    df_sorted['start:timestamp'] = df_sorted['start:timestamp'].fillna(df_sorted['time:timestamp'])
    return df_sorted


def compute_next_temporal_features(prev_day_of_week, prev_time_of_day, avg_execution_time):
    prev_time_seconds = prev_time_of_day * 86400
    new_time_seconds = prev_time_seconds + avg_execution_time
    days_passed = int(new_time_seconds // 86400)
    new_time_seconds_in_day = new_time_seconds % 86400
    new_day_of_week = (prev_day_of_week + days_passed) % 7
    new_time_of_day = new_time_seconds_in_day / 86400
    time_delta = avg_execution_time
    return {
        'day_of_week': new_day_of_week,
        'time_of_day': new_time_of_day,
        'time_delta': time_delta
    }

def create_new_feat_vec(original_feat_vec, cf_feat_vec, new_value, index, seq_len, features, avg_execution_time, scalers):
    new_feat_vec = original_feat_vec.clone()
    new_feat_vec[index] = new_value
    new_cf_feat_vec = cf_feat_vec.clone()
    new_cf_feat_vec[index] = new_value
    new_temp_features = compute_next_temporal_features(
        prev_day_of_week=original_feat_vec[(features.index('day_of_week') * seq_len) + index - 1].item(),
        prev_time_of_day=scalers['time_of_day'].inverse_transform([[original_feat_vec[(features.index('time_of_day') * seq_len) + index - 1].item()]])[0][0],
        avg_execution_time=avg_execution_time
    )

    for feature in features:
        if feature != 'activity' and feature != 'org:resource' and feature not in ['day_of_week', 'time_of_day', 'time_delta']:
            index_feat = features.index(feature) * seq_len + index
            new_feat_vec[index_feat] = original_feat_vec[index_feat-1]
            new_cf_feat_vec[index_feat] = cf_feat_vec[index_feat-1]
        elif feature == 'day_of_week':
            index_feat = features.index('day_of_week') * seq_len + index
            new_feat_vec[index_feat] = new_temp_features['day_of_week']
            new_cf_feat_vec[index_feat] = new_temp_features['day_of_week']
        elif feature == 'time_of_day':
            index_feat = features.index('time_of_day') * seq_len + index
            scaled_time_of_day = scalers['time_of_day'].transform([[new_temp_features['time_of_day']]])[0][0]
            new_feat_vec[index_feat] = scaled_time_of_day
            new_cf_feat_vec[index_feat] = scaled_time_of_day
        elif feature == 'time_delta':
            index_feat = features.index('time_delta') * seq_len + index
            scaled_time_delta = scalers['time_delta'].transform([[new_temp_features['time_delta']]])[0][0]
            new_feat_vec[index_feat] = scaled_time_delta
            new_cf_feat_vec[index_feat] = scaled_time_delta
        
    return new_feat_vec, new_cf_feat_vec

def recommend_act_loader(test_loader, ts, model, activity_mapping, seq_len, features, avg_exec_time_dict, scalers, device, dataset_name, model_name, save_to_file=False):
    act_kpi_og_final_results = []
    act_kpi_cf_final_results = []
    with torch.no_grad():
        for x, x_cf, y, case_id in test_loader:
            for i in range(len(x)):
                index = (x[i][:seq_len] == 0).nonzero(as_tuple=True)[0]
                prefix = [activity_mapping[int(act)] for act in x[i][:seq_len][x[i][:seq_len] != 0].tolist()]
                next_acts = ts.get_val(prefix)
                act_kpi_list = []
                act_kpi_list_cf = []
                for act in next_acts:
                    act_int = [k for k, v in activity_mapping.items() if v == act][0]
                    if len(index) > 0:
                        new_x_og, new_x_cf = create_new_feat_vec(x[i], x_cf[i], act_int, index[0], seq_len, features, avg_exec_time_dict.get(act, 0), scalers)
                        prediction_og = model(new_x_og.unsqueeze(0).to(device))
                        act_kpi_list.append((case_id[i].item(), act, max(0, scalers['remaining_time'].inverse_transform([[prediction_og.item()]])[0][0])))
                        prediction_cf = model(new_x_cf.unsqueeze(0).to(device))
                        act_kpi_list_cf.append((case_id[i].item(), act, max(0, scalers['remaining_time'].inverse_transform([[prediction_cf.item()]])[0][0])))
                act_kpi_list.sort(key=lambda x: x[2])
                act_kpi_list_cf.sort(key=lambda x: x[2])
                act_kpi_og_final_results.append(act_kpi_list[0])
                act_kpi_cf_final_results.append(act_kpi_list_cf[0])
    if save_to_file:
        with open(f"experiments/{dataset_name}/{model_name}_recommended_activities.csv", "w") as f:
            f.write("case:concept:name,concept:name,kpi\n")
            for og in act_kpi_og_final_results:
                f.write(f"{og[0]},{og[1]},{og[2]}\n")
        with open(f"experiments/{dataset_name}/{model_name}_recommended_activities_counterfactual.csv", "w") as f:
            f.write("case:concept:name,concept:name,kpi\n")
            for cf in act_kpi_cf_final_results:
                f.write(f"{cf[0]},{cf[1]},{cf[2]}\n")
            
    return act_kpi_og_final_results, act_kpi_cf_final_results

def compute_cf_fairness(act_kpi_list, act_kpi_list_cf):
    counterfactual_fairness = 0
    for i in range(len(act_kpi_list)):
        if act_kpi_list[i][1] == act_kpi_list_cf[i][1]:
            counterfactual_fairness += 1
    counterfactual_fairness = counterfactual_fairness / len(act_kpi_list)
    return counterfactual_fairness

def compute_activity_resource_probabilities(
    event_log,
    activity_col='concept:name',
    resource_col='org:resource'):
    """
    Compute resource probability distributions per activity.

    Returns
    -------
    dict
        Mapping:
        {
            <activity>: [(<resource_1>, <prob_1>), (<resource_2>, <prob_2>), ...]
        }
    """
    if activity_col not in event_log.columns:
        raise ValueError(f"Column '{activity_col}' not found in event_log.")
    if resource_col not in event_log.columns:
        raise ValueError(f"Column '{resource_col}' not found in event_log.")

    df = event_log[[activity_col, resource_col]].dropna()
    if df.empty:
        return {}

    # Frequency table and normalized probabilities per activity.
    counts = (
        df.groupby([activity_col, resource_col])
        .size()
        .rename('count')
        .reset_index()
    )
    counts['probability'] = counts.groupby(activity_col)['count'].transform(
        lambda x: x / x.sum()
    )

    distributions = {}
    for activity, group in counts.groupby(activity_col):
        group_sorted = group.sort_values('probability', ascending=False)
        distributions[activity] = [
            (row[resource_col], float(row['probability']))
            for _, row in group_sorted.iterrows()
        ]

    return distributions