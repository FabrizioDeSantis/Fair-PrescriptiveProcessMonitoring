import torch
from collections import Counter

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

def create_single_new_feat_vec(original_feat_vec, cf_feat_vec, new_value, index, seq_len, features, avg_execution_time, scalers):
    new_feat_vec = original_feat_vec.clone()
    new_feat_vec[index] = new_value
    new_cf_feat_vec = cf_feat_vec.clone()
    new_cf_feat_vec[index] = new_value
    if 'day_of_week' in features and 'time_of_day' in features and 'time_delta' in features:
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
    actv_recc = []
    count_A_CANCELLED = 0
    threshold = int(0.1 * len(test_loader.dataset))
    with torch.no_grad():
        for x, x_cf, y, case_id in test_loader:
            x_cf = x_cf.squeeze(1)
            for i in range(len(x)):
                index = (x[i][:seq_len] == 0).nonzero(as_tuple=True)[0]
                prefix = [activity_mapping[int(act)] for act in x[i][:seq_len][x[i][:seq_len] != 0].tolist()]
                next_acts = ts.get_val(prefix)
                if isinstance(next_acts, list):
                    actv_recc.append(len(next_acts))
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
                    recommended_og = act_kpi_list[0]
                    recommended_cf = act_kpi_list_cf[0]
                    if recommended_og[1] == "A_CANCELLED":
                        if count_A_CANCELLED >= threshold:
                            recommended_og = act_kpi_list[1] if len(act_kpi_list) > 1 else act_kpi_list[0]
                        else:
                            count_A_CANCELLED += 1
                    if recommended_cf[1] == "A_CANCELLED":
                        # Use the same threshold logic for counterfactual
                        if count_A_CANCELLED >= threshold:
                            recommended_cf = act_kpi_list_cf[1] if len(act_kpi_list_cf) > 1 else act_kpi_list_cf[0]
                    act_kpi_og_final_results.append(recommended_og)
                    act_kpi_cf_final_results.append(recommended_cf)
    if save_to_file:
        with open(f"{dataset_name}_{model_name}_recommended_activities.csv", "w") as f:
            f.write("case:concept:name,concept:name,kpi\n")
            for og in act_kpi_og_final_results:
                f.write(f"{og[0]},{og[1]},{og[2]}\n")
        with open(f"{dataset_name}_{model_name}_recommended_activities_counterfactual.csv", "w") as f:
            f.write("case:concept:name,concept:name,kpi\n")
            for cf in act_kpi_cf_final_results:
                f.write(f"{cf[0]},{cf[1]},{cf[2]}\n")
    actv_recc_counts = Counter(actv_recc)
    print("Activity recommendation counts:", dict(actv_recc_counts))
    return act_kpi_og_final_results, act_kpi_cf_final_results

def compute_cf_fairness(act_kpi_list, act_kpi_list_cf):
    counterfactual_fairness = 0
    for i in range(len(act_kpi_list)):
        if act_kpi_list[i][1] == act_kpi_list_cf[i][1]:
            counterfactual_fairness += 1
    counterfactual_fairness = counterfactual_fairness / len(act_kpi_list)
    return counterfactual_fairness