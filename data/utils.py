import pandas as pd
import numpy as np
import datetime
import torch

def remaining_time_helper_func(df, case_id_column):
    case_id = case_id_column
    event_time = "time:timestamp"
    unique_cases = df[case_id].unique()
    remaining_times = []
    df[event_time] = df[event_time].astype(str)
    for _, case in enumerate(unique_cases):
        time = df[df[case_id] == case][event_time].str[:19].to_list()
        for i in range(0, len(time)):
            time_stamp = time[i]
            tcc = datetime.datetime.strptime(time[-1], '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(time_stamp, '%Y-%m-%d %H:%M:%S')
            tcc = tcc.total_seconds() / 86400
            remaining_times.append(tcc)
    df["remaining_time"] = remaining_times
    return df

def create_temporal_features(df):
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], format="mixed")
    df['day_of_week'] = df['time:timestamp'].dt.dayofweek
    df['time_of_day'] = (
        (df['time:timestamp'].dt.hour * 3600) + 
        (df['time:timestamp'].dt.minute * 60) + 
        (df['time:timestamp'].dt.second)
    ) / 86400
    df["time_delta"] = df.groupby("case:concept:name")["time:timestamp"].diff().dt.total_seconds().fillna(0)
    return df

def compute_average_execution_time(event_log, case_col, activity_col, timestamp_col):
    
    df = event_log.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], format="mixed")
    
    df = df.sort_values([case_col, timestamp_col]).reset_index(drop=True)
    
    df['prev_timestamp'] = df.groupby(case_col)[timestamp_col].shift(1)
    df['execution_time'] = df[timestamp_col] - df['prev_timestamp']
    
    df['execution_time_seconds'] = df['execution_time'].dt.total_seconds()
    
    df_with_exec_time = df[df['execution_time_seconds'].notna()].copy()
    
    avg_times = df_with_exec_time.groupby(activity_col).agg(
        avg_execution_time=('execution_time_seconds', 'mean'),
        count=('execution_time_seconds', 'count')
    ).reset_index()
    
    avg_times = avg_times.sort_values('avg_execution_time', ascending=False)
    
    result_dict = dict(zip(avg_times[activity_col], avg_times['avg_execution_time']))
    
    return result_dict

def compute_cf_mse(predictions, predictions_cf):
    predictions = np.array(predictions)
    predictions_cf = np.array(predictions_cf)
    mse = np.mean((predictions - predictions_cf) ** 2)
    return mse

if __name__ == "__main__":
    data = pd.read_pickle("bpi12.pkl")
    avg_exec_times = compute_average_execution_time(data, case_col="case_id", activity_col="activity", timestamp_col="time:timestamp")
    for activity, avg_time in avg_exec_times.items():
        print(f"Activity: {activity}, Average Execution Time (seconds): {avg_time:.2f}")