import pandas as pd
import numpy as np
import datetime

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

if __name__ == "__main__":
    data = pd.read_csv('bpi12.csv')
    data = create_temporal_features(data)
    data.to_csv("bpi12_enhanced.csv", index=False)
    print(data.head())