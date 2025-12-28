# Transition System Documentation

## Overview

The transition system is a data structure that maps sequences of activities (traces) to their possible next activities. It is implemented as a hash map that can be used for process prediction, anomaly detection, and recommendation systems.

## Creating a Transition System

### Function: `fill_hashmap()`

```python
from transition_system import hash_maps

traces_hash = hash_maps.fill_hashmap(
    X_train=X_train,
    case_id_name="case_id",
    activity_name="activity",
    thrs=0,
    abstraction="sequence",
    k=None
)
```

### Parameters

#### `X_train` (required)
- **Type**: `pandas.DataFrame`
- **Description**: Event log data containing cases and activities
- **Required columns**: Must include the columns specified by `case_id_name` and `activity_name`

#### `case_id_name` (required)
- **Type**: `str`
- **Description**: Name of the column containing case identifiers
- **Examples**: `"case_id"`, `"case:concept:name"`

#### `activity_name` (required)
- **Type**: `str`
- **Description**: Name of the column containing activity names
- **Examples**: `"activity"`, `"concept:name"`

#### `thrs` (optional, default: `0.2`)
- **Type**: `float`
- **Description**: Threshold for frequency filtering of traces
  - `thrs = 0`: Include ALL traces (no filtering)
  - `thrs = 0.2`: Include top 20% most frequent traces (Pareto principle)
  - `thrs > 0`: Include top `thrs * 100%` most frequent traces
- **Use case**: Use `thrs=0` to include all patterns, or higher values to focus on common patterns

#### `abstraction` (optional, default: `"sequence"`)
- **Type**: `str` or `False`
- **Values**: 
  - `"sequence"` or `False`: Order matters (default behavior)
  - `"set"`: Order doesn't matter (treats `[A, B]` and `[B, A]` as the same)
- **Description**: Determines how activity sequences are treated as keys
- **Example**:
  - `abstraction="sequence"`: `get_val(['A', 'B'])` ≠ `get_val(['B', 'A'])`
  - `abstraction="set"`: `get_val(['A', 'B'])` = `get_val(['B', 'A'])`

#### `k` (optional, default: `None`)
- **Type**: `int` or `None`
- **Description**: Number of last activities to use as key
  - `k = None`: Use full sequence as key (default)
  - `k = 1`: Use only the last 1 activity (first-order Markov model)
  - `k = 2`: Use only the last 2 activities (second-order Markov model)
  - `k = n`: Use only the last n activities (n-th order Markov model)
- **Use case**: 
  - Reduces memory usage when `k` is small
  - Creates a k-order Markov model
  - Useful when only recent history matters
- **Example**:
  - With `k=2`, sequence `[A, B, C, D]` uses `[C, D]` as the key
  - With `k=None`, sequence `[A, B, C, D]` uses `[A, B, C, D]` as the key

### Return Value

- **Type**: `HashTable` object
- **Description**: A hash table containing the transition system
- **Attributes**:
  - `abstraction`: The abstraction mode used (`"sequence"` or `"set"`)

## Using the Transition System

### Method: `get_val(key)`

Query the transition system to get possible next activities for a given sequence.

```python
# Get possible next activities
next_activities = traces_hash.get_val(['A', 'B'])
# Returns: ['C', 'D'] or "No record found"
```

#### Parameters

- **`key`**: `list` or `str`
  - List of activities (e.g., `['A', 'B']`)
  - String representation (e.g., `"A, B"`)
  - **Important**: When using `abstraction="set"`, the order doesn't matter
  - **Important**: When using `k`, provide the last k activities (or full sequence if shorter)

#### Return Value

- **Type**: `list` or `str`
  - List of possible next activities if found
  - `"No record found"` if the key doesn't exist

### Examples

#### Example 1: Basic Usage

```python
import pandas as pd
from transition_system import hash_maps

# Create sample data
data = pd.DataFrame({
    'case_id': [1, 1, 1, 2, 2, 2],
    'activity': ['A', 'B', 'C', 'A', 'B', 'D']
})

# Create transition system
traces_hash = hash_maps.fill_hashmap(
    X_train=data,
    case_id_name='case_id',
    activity_name='activity',
    thrs=0,
    abstraction='sequence',
    k=None
)

# Query the system
print(traces_hash.get_val(['A']))        # ['B']
print(traces_hash.get_val(['A', 'B']))   # ['C', 'D']
print(traces_hash.get_val(['X']))        # "No record found"
```

#### Example 2: Using k Parameter

```python
# Create transition system with k=2 (only last 2 activities matter)
traces_hash = hash_maps.fill_hashmap(
    X_train=data,
    case_id_name='case_id',
    activity_name='activity',
    thrs=0,
    abstraction='sequence',
    k=2
)

# Query with last 2 activities
print(traces_hash.get_val(['A', 'B']))   # ['C', 'D']
print(traces_hash.get_val(['B', 'C']))   # Next activities after B->C
```

#### Example 3: Using Set Abstraction

```python
# Create transition system with set abstraction (order doesn't matter)
traces_hash = hash_maps.fill_hashmap(
    X_train=data,
    case_id_name='case_id',
    activity_name='activity',
    thrs=0,
    abstraction='set',
    k=None
)

# Both queries return the same result
print(traces_hash.get_val(['A', 'B']))   # ['C', 'D']
print(traces_hash.get_val(['B', 'A']))   # ['C', 'D'] (same as above)
```

## Loading a Saved Transition System

Transition systems are saved with the following naming convention:

```
transition_system/variables/{experiment_name}/transition_system_{abstraction}[_k{k}].pkl
```

### Examples:

- `transition_system_set.pkl` - Set abstraction, full sequence (k=None)
- `transition_system_sequence.pkl` - Sequence abstraction, full sequence (k=None)
- `transition_system_set_k2.pkl` - Set abstraction, k=2
- `transition_system_sequence_k1.pkl` - Sequence abstraction, k=1

### Loading Code:

```python
import pickle

# Load transition system
experiment_name = "bpi12_kappel"
abstraction_mode = "set"
k = 2  # or None

if k is not None:
    filename = f'transition_system/variables/{experiment_name}/transition_system_{abstraction_mode}_k{k}.pkl'
else:
    filename = f'transition_system/variables/{experiment_name}/transition_system_{abstraction_mode}.pkl'

traces_hash = pickle.load(open(filename, 'rb'))
```

## Use Cases

1. **Process Prediction**: Given a partial trace, predict next possible activities
2. **Anomaly Detection**: Identify traces that don't follow common patterns
3. **Recommendation System**: Suggest next steps based on current state
4. **Process Mining**: Analyze process flow and transitions

## Notes

- The hash table uses **chaining** for collision resolution
- Keys are stored as **strings** (comma-separated activity lists)
- Values are **lists** of possible next activities
- Empty lists indicate **end of trace** (no more activities possible)
- When using `k`, make sure to query with the appropriate number of activities
