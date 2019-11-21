import os
from pathlib import Path
import shutil
import numpy as np
from numpy import random as npr
import pandas as pd
from IPython.display import display
import yaml
from tqdm import tqdm_notebook as tqdm
#test
# Hyperparams
# data_dir_str = '../../data'
data_dir_str = '/Users/pangda/predicting_generalization/alldata'
data_path_str = 'Grp13_conv_random'
new_path_str = 'split_set'
splits = [
    ('train', .8),
    ('val', .1),
    ('test', .1)
]

# seed
npr.seed(523324)

# build paths
base_path = Path(data_dir_str)
save_path = base_path / Path(new_path_str).expanduser()
data_path = base_path /Path(data_path_str).expanduser()
type_dirs = [ f for f in data_path.iterdir() if f.is_dir() ]

# determine cumulative splits and validate split sizes
total = 0.
splits_total = []

for split_name, prop in splits:
    splits_total.append((split_name, total, total + prop))
    total += prop

assert abs(splits_total[-1][-1] - 1.) < 1e-10, 'Invalid split (does not sum to 1)'

# randomly split all models
split_mdirs = {split_name: [] for split_name, _ in splits}

def process_type(model_dirs, rand_idx, split_name):
    """
    Add all models in `model_dirs` with index in
    `rand_idx` to split `split name`.
    """
    for src_idx in rand_idx:
        suffix = f'_{src_idx}'
        matches = [mdir for mdir in model_dirs if str(mdir)[-len(suffix):] == suffix]

        assert len(matches) == 1, f'Invalid source model name format, {len(matches)} matches found'

        src_path = matches[0]
        split_mdirs[split_name].append(src_path)

# iterate over each model type
for type_dir in type_dirs:
    # get all model dirs for type and randomly partition into splits
    model_dirs = [ f for f in type_dir.iterdir() if f.is_dir() ]
    rand_idx = npr.permutation(len(model_dirs))

    for split_name, start, stop in splits_total:
        process_type(model_dirs,
                     rand_idx[int(len(rand_idx) * start): int(len(rand_idx) * stop)],
                     split_name)

# Randomly reorder models in each split, copy the model to the new directory, and
#    record the mapping
mapping = []

for split_name, model_dirs in split_mdirs.items():
    rand_idx = npr.permutation(len(model_dirs))

    for dst_idx, src_idx in enumerate(rand_idx):
        src_path = model_dirs[src_idx]
        dst_path = save_path / split_name / f'model_{dst_idx}'
        shutil.copytree(src_path, dst_path)
        mapping.append((split_name, dst_idx, os.path.join(*src_path.parts[-2:])))

# save the mapping to a CSV
df = pd.DataFrame(mapping, columns=['Split Name', 'Index', 'Source File'])
df.to_csv(os.path.join(data_dir_str, 'mapping.csv'), index=False)

# fill defaults

metadata_defaults = {
    'batch_size_train': 512,
    'batch_size_test': 1024,
    'batch_size_val': 512
}

other_defaults = {}
stats_defaults = {}

def fill_missing_yaml(path, defaults):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    for key, val in defaults.items():
        if key not in data:
            data[key] = val
    with open(path, 'w') as f:
        f.write(yaml.dump(data, default_flow_style=False))

def fill_missing(row):
    path = os.path.join(data_dir_str, new_path_str, row['Split Name'], f"model_{row['Index']}")
    fill_missing_yaml(os.path.join(path, 'meta_data.yml'), metadata_defaults)
    ## fill other_data.yml
    ## fill param_stats.yml

for i, row in tqdm(df.iterrows()):
    fill_missing(row)


# Initial Dataset Validation

print('Proportion by Split:')
display(df.groupby('Split Name')['Index'].count() / len(df))

print('\nUnique Source Files by Split:')
display(df.groupby('Split Name')['Source File'].nunique())

assert df['Source File'].unique().shape[0] == len(df), 'Every source file is not unique'
