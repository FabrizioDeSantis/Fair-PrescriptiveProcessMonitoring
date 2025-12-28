#%%
import pickle
import time
import random
import os
random.seed(1618)

# Converter
import pm4py

#My functions (only IO needed for first block)
from IO import read, folders, create_folders

experiment_name = "bpi12_kappel"

# Read the log with the IO.read function
# Try CSV first, then pickle if CSV doesn't exist
data_path_csv = f'../data/{experiment_name}.csv'
data_path_pkl = f'../data/{experiment_name}.pkl'

if os.path.exists(data_path_csv):
    log = read(data_path_csv)
    print(f"Successfully read CSV file: {data_path_csv}")
elif os.path.exists(data_path_pkl):
    log = read(data_path_pkl)
    print(f"Successfully read pickle file: {data_path_pkl}")
else:
    raise FileNotFoundError(f"Neither {data_path_csv} nor {data_path_pkl} found")

# Keep original DataFrame for transition system processing
X_train = log.copy()

# Define column names for the transition system
# Check what columns we have and set appropriate names
if 'case_id' in X_train.columns:
    case_id_name = 'case_id'
elif 'case:concept:name' in X_train.columns:
    case_id_name = 'case:concept:name'
else:
    raise ValueError("Could not find case ID column. Available columns: " + str(list(X_train.columns)))

if 'activity' in X_train.columns:
    activity_name = 'activity'
elif 'concept:name' in X_train.columns:
    activity_name = 'concept:name'
else:
    raise ValueError("Could not find activity column. Available columns: " + str(list(X_train.columns)))

print(f"Using case_id_name: {case_id_name}, activity_name: {activity_name}")

# Check if we need to rename columns for pm4py compatibility
# pm4py expects 'case:concept:name' but the data might have 'case_id'
log_pm4py = log.copy()
if 'case_id' in log_pm4py.columns and 'case:concept:name' not in log_pm4py.columns:
    log_pm4py = log_pm4py.rename(columns={'case_id': 'case:concept:name'})
    print("Renamed 'case_id' to 'case:concept:name' for pm4py compatibility")

# Check if activity column exists (pm4py needs 'concept:name' or 'activity')
if 'activity' in log_pm4py.columns and 'concept:name' not in log_pm4py.columns:
    log_pm4py = log_pm4py.rename(columns={'activity': 'concept:name'})
    print("Renamed 'activity' to 'concept:name' for pm4py compatibility")

log = pm4py.convert_to_event_log(log_pm4py)
print(f"Successfully converted to event log. Number of traces: {len(log)}")

#%%
# Import functions needed for subsequent blocks
import hash_maps
import utils

# # Get available resources from the event log or DataFrame
# if 'org:resource' in X_train.columns:
#     available_resources_list = list(X_train['org:resource'].dropna().unique())
#     print(f"Found {len(available_resources_list)} unique resources")
# else:
#     # Try to get from pm4py event log
#     try:
#         available_resources_list = list(pm4py.get_event_attribute_values(log, "org:resource").keys())
#         print(f"Found {len(available_resources_list)} unique resources from event log")
#     except:
#         available_resources_list = []
#         print("Warning: Could not find resources. Using empty list.")

# Get activity list
activity_list = list(X_train[activity_name].unique())
print(f"Found {len(activity_list)} unique activities")

# roles = pm4py.discover_organizational_roles(log)

# """ Mi sto creando un dizionario ruoli, puoi skippare"""
# act_role_dict = dict()
# cases_list = list(df_rec['case:concept:name'].unique())
# for act in activity_list:
#     for idx in range(len(roles)):
#         if act in roles[idx][0]:
#             act_role_dict[act] = list(roles[idx][1].keys())
# """ Fin qui """

# QUESTO È IL TRANSITION SYSTEM
traces_hash = hash_maps.fill_hashmap(X_train=X_train, case_id_name=case_id_name, activity_name=activity_name, thrs=0)
# traces_hash = pickle.load(open('traces_hash.pkl', 'rb'))

print('Hash-map created')
# print('Analyze variables...')
# quantitative_vars, qualitative_trace_vars, qualitative_vars = utils.variable_type_analysis(X_train, case_id_name,
#                                                                                            activity_name)
# # warnings.filterwarnings("ignore")
# print('Variable analysis done')
# # Create variables directory if it doesn't exist
# variables_dir = f'variables/{experiment_name}'
# os.makedirs(variables_dir, exist_ok=True)

# pickle.dump(quantitative_vars, open(f'{variables_dir}/quantitative_vars.pkl', 'wb'))
# pickle.dump(qualitative_vars, open(f'{variables_dir}/qualitative_vars.pkl', 'wb'))
# pickle.dump(traces_hash, open(f'{variables_dir}/traces_hash.pkl', 'wb'))
# print(f"Saved results to {variables_dir}/")

# """ Implementazione per raccomdandare risorse"""
# #Filter the resources for keeping just the active resources, IT ALSO REMOVES MISSING
# if len(available_resources_list) > 0:
#     try:
#         available_resources_list = utils.filter_resources_availability(available_resources_list, p=.75)
#         if 'missing' in available_resources_list: 
#             available_resources_list.remove('missing')
#         print(f"Filtered resources: {len(available_resources_list)} remaining")
#     except Exception as e:
#         print(f"Warning: Could not filter resources: {e}")
# else:
#     print("Warning: No resources available to filter")
# """Finisce qui"""


# %%
