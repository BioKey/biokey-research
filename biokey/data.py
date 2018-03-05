import pandas as pd
import numpy as np
from tqdm import tqdm_notebook, tqdm
from sqlalchemy import create_engine
import os
from .keys import key_table



directory = os.path.expanduser("~/biokey_data")
if not os.path.exists(directory):
    os.makedirs(directory)

def process_user(user_strokes):
    # -------------
    # Output Strokes
    # -------------
    user = user_strokes.user_id.iloc[0]
    user_directory = os.path.expanduser('{0}/users/{1}'.format(directory, user))
    if not os.path.exists(user_directory):
        os.makedirs(user_directory)
    user_strokes.drop('user_id', 1).to_csv('{0}/strokes.csv'.format(user_directory), index=False)
    # -------------
    # Output dwells
    # -------------
    dwells = pd.DataFrame()
    for key, strokes in user_strokes.groupby('key_code'):
        # Eliminate non duplicate strokes
        strokes = strokes.loc[(strokes.direction != strokes.direction.shift(1))]
        # Split up and down actions
        down = strokes.loc[strokes.direction == 'd'].reset_index(drop=True)
        up = strokes.loc[strokes.direction == 'u'].reset_index(drop=True)
        # Filter non-matching strokes
        number = np.min([down.count(), up.count()])
        down = down.head(number)
        up = up.head(number)
        # Add to DataFrame
        key_params = key_table[key] if key in key_table else None
        dwells = dwells.append(pd.DataFrame({
            'key_code': key,
            'key_enum': key_params['enum'] if key_params is not None else None,
            'key': key_params['key'] if key_params is not None else None,
            'down': down.key_time,
            'up': up.key_time,
            'dwell': up.key_time - down.key_time
        }), ignore_index=True)
    # Output user's dwells to csv
    dwells = dwells.sort_values('down')[['key_code', 'key_enum', 'key', 'down', 'up', 'dwell']].reset_index(drop=True)
    dwells.to_csv('{0}/dwells.csv'.format(user_directory), index=False)
    # -------------
    # Output flights
    # -------------
    flights = pd.DataFrame({
        'key_orig': dwells.key_code,
        'key_dest': dwells.shift(-1).key_code,
        'orig_down': dwells.down,
        'orig_up': dwells.up,
        'dest_down': dwells.shift(-1).down,
        'dest_up': dwells.shift(-1).up,
        'interval': dwells.shift(-1).down - dwells.up,
        'down_to_down': dwells.shift(-1).down - dwells.down,
        'up_to_up': dwells.shift(-1).up - dwells.up,
        'duration': dwells.shift(-1).up - dwells.down
    }).sort_values('orig_down')[['key_orig', 'key_dest', 'orig_down', 'orig_up', 'dest_down', 'dest_up', 'interval', 'down_to_down', 'up_to_up', 'duration']].reset_index(drop=True)
    flights.to_csv('{0}/flights.csv'.format(user_directory), index=False)
    # Prepare to return
    dwells['user_id'] = user
    flights['user_id'] = user
    return pd.Series({'dwells': dwells, 'flights': flights})

def process(strokes):
    try:
        get_ipython
        tqdm_notebook().pandas(desc="Loading Data")
    except:
        tqdm().pandas(desc="Loading Data")
    results = strokes.groupby('user_id').progress_apply(process_user)
    dwells = pd.concat([i for i in results.dwells], ignore_index=False).sort_values('down')
    # Remove negative dwells and upper outliers
    dwells = dwells.loc[(dwells.dwell > 0) & (dwells.dwell < dwells.dwell.quantile(0.999))]
    flights = pd.concat([i for i in results.flights], ignore_index=False).sort_values('orig_down')
    return pd.Series({'dwells': dwells, 'flights': flights})




def user_sequence_label(dwells, thresh):
    dwells = dwells.sort_values('down')
    dwells['Interval'] = (dwells.down - dwells.up.shift(1)).fillna(0)
    dividers = dwells.Interval > thresh
    dwells['Sequence'] = dividers.cumsum()
    return dwells

def user_sequence_split(dwells, thresh):
    return user_sequence_label(dwells, thresh).groupby('Sequence').apply(lambda x: pd.Series({
        'start': x.down.min(), 
        'finish': x.up.max(), 
        'duration': x.up.max() - x.down.min()
    }))

def get_sequences(dwells, thresh):
    return dwells.groupby('user_id').progress_apply(lambda x: user_sequence_split(x, thresh)).reset_index()

def get_user_set(selected_user, sampled_seqs, seqs, all_dwells):
    # Select the user's sequences from their sampled sequences and calculate the range of time they cover
    users_seqs = sampled_seqs.loc[sampled_seqs.user_id == selected_user].sort_values('start')
    ranges_to_fill = pd.DataFrame({'start': users_seqs.finish, 'finish': users_seqs.start.shift(-1)})
    # Select all other sequences that fit within the spaces from other users
    non_user_seqs = sampled_seqs.loc[sampled_seqs.user_id != selected_user].sort_values('start')
    to_include = pd.concat([non_user_seqs.loc[(non_user_seqs.start > x.start) & (non_user_seqs.finish < x.finish)] for i, x in ranges_to_fill.iterrows()], ignore_index=True)
    # Merge both values and assure no overlap
    set_sequences = pd.concat([users_seqs, to_include], ignore_index=True).sort_values('start').reset_index()
    set_sequences = set_sequences.loc[set_sequences.start.shift(-1) - set_sequences.finish > 0]
    # Select dwells chosen as part of selected training sequences
    set_indexes = set_sequences[['user_id', 'Sequence']].set_index(['user_id', 'Sequence']).index
    set_dwells = all_dwells.copy().set_index(['user_id', 'Sequence']).loc[set_indexes].reset_index().sort_values('down')
    set_dwells['is_user'] = set_dwells.user_id == selected_user
    return set_dwells[['key_code', 'key_enum', 'key', 'down', 'up', 'dwell', 'is_user']]

def get_user_split_sets(selected_user, seqs, all_dwells):
    # Select random sample of each user's sequence
    selected_seqs = seqs.groupby('user_id').apply(lambda x: x.sample(frac=0.5).set_index('Sequence')).index
    # Select the actual sequence from the indexes and generate a train set with it
    sampled_seqs = seqs.set_index(['user_id', 'Sequence']).loc[selected_seqs].sort_values('start').reset_index()
    train = get_user_set(selected_user, sampled_seqs, seqs, all_dwells)
    # Select the opposite sequences and generate a test set with it
    sampled_seqs = seqs.set_index(['user_id', 'Sequence']).drop(selected_seqs).sort_values('start').reset_index()
    test = get_user_set(selected_user, sampled_seqs, seqs, all_dwells)
    # Return both sets
    return {'train': train, 'test': test}


class UserDataset:
    def __init__(self, datasets):
        self.test = datasets['test']
        self.train = datasets['train']

class DataInterface:
    
    def __init__(self, credentials, limit=None, force_load= False, force_process=False):
        # Load in data from cache or server
        print("Loading Data")
        try: 
            if force_load:
                raise Exception('Force Reload')
            print("\t- Attempting cache load")
            df = pd.read_csv('{0}/strokes.csv'.format(directory)).sort_values('key_time')
            print("\t- Loaded strokes from cache")
        except:
            print("\t- Missed stroke cache")
            df = pd.read_sql_query('SELECT * FROM strokes' + (('LIMIT %s' % limit) if limit is not None else ''), create_engine(credentials))
            df.to_csv('{0}/strokes.csv'.format(directory), index=False)
            print("\t- Done and cached for later")
        self.strokes = df
        # Process data
        print("Processing Data")
        try:
            if force_load or force_process:
                raise Exception('Force Reload')
            print("\t- Attempting cache load")
            self._dwells = pd.read_csv('{0}/dwells.csv'.format(directory)).sort_values('down')
            self._flights = pd.read_csv('{0}/flights.csv'.format(directory)).sort_values('orig_down')
            print("\t- Loaded dwell and flight from cache")
        except:
            print("\t- Missed dwell and flight cache")
            results = process(df)
            self._dwells = results.dwells
            self._flights = results.flights
            self._dwells.to_csv('{0}/dwells.csv'.format(directory), index=False)
            self._flights.to_csv('{0}/flights.csv'.format(directory), index=False)
            print("\t- Generating train and test sets")
            self.generate_sets()
            print("\t- Done and cached for later")
        print("Done Loading\n")

    def get_dwells(self, user=None):
        if user is not None:
            return self._dwells.loc[self._dwells.user_id == user].reset_index(drop=True)
        return self._dwells.reset_index(drop=True)

    def get_flights(self, user=None):
        if user is not None:
            return self._flights.loc[self._flights.user_id == user].reset_index(drop=True)
        return self._flights.reset_index(drop=True)
    
    def get_users(self):
        return self._dwells.user_id.value_counts().index

    def generate_sets(self, thresh = 60*1000):
        try:
            get_ipython
            tqdm_notebook().pandas(desc="Generating Sets")
        except:
            tqdm().pandas(desc="Generating Sets")
        all_dwells = self._dwells
        seqs = get_sequences(all_dwells, thresh)
        # Add in sequence numbers
        all_dwells = all_dwells.groupby('user_id').apply(lambda x: user_sequence_label(x, thresh)).reset_index(drop=True).sort_values('down')
        users =  pd.Series(all_dwells.user_id.unique())
        datasets = pd.DataFrame({
            'user': users, 
            'value': users.progress_apply(lambda user: get_user_split_sets(user, seqs, all_dwells))
        }).set_index('user').value.to_dict()
        # datasets = {user: get_user_split_sets(user, seqs, all_dwells) for user in all_dwells.user_id.unique()}
        for user in datasets:
            user_directory = os.path.expanduser('{0}/users/{1}'.format(directory, user))
            if not os.path.exists(user_directory):
                os.makedirs(user_directory)
            datasets[user]['train'].to_csv('{0}/train.csv'.format(user_directory), index=False)
            datasets[user]['test'].to_csv('{0}/test.csv'.format(user_directory), index=False)
        return datasets

    def get_user_sets(self, user):
        datasets = {}
        user_directory = os.path.expanduser('{0}/users/{1}'.format(directory, user))
        try:
            datasets['train'] = pd.read_csv('{0}/train.csv'.format(user_directory))
            datasets['test'] = pd.read_csv('{0}/test.csv'.format(user_directory))
        except:
            print("Missed Cache - Generating Sets")
            datasets = self.generate_sets()[user]
        return UserDataset(datasets)
    
    def get_all_sets(self):
        users = self.get_users()
        return {u: self.get_user_sets(u) for u in users}
