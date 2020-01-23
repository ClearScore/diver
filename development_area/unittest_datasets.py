import numpy as np
import pandas as pd

def _generate_X_train_1():

    # Generate X_train dataframe with missing values

    # Initialise parameters
    n_rows = 10
    nominal_categories = {'stanage': 0.3, 'burbage': 0.2, 'almscliff': 0.2, 'froggatt': 0.15, 'blacknor':0.15}

    # Generate dataframe
    np.random.seed(0)
    X_train = pd.DataFrame({
        'numeric_1': np.random.choice(100, n_rows), 
        'numeric_2': np.random.choice(10, n_rows, replace=False), 
        'numeric_3': np.random.choice(22, n_rows, replace=False), 
        'numeric_4': np.random.choice(5, n_rows), 
        'nominal': np.random.choice(list(nominal_categories.keys()), n_rows, replace=True, p=list(nominal_categories.values())), 
    })
    np.random.seed(0)
    X_train = X_train.mask((np.random.random(size=X_train.shape) > 0.75))

    # Add boolean (this doesn't work before mask applied as this turns bool dtype to float if NaNs exist)
    bool_elements = {True: 0.6, False: 0.4}
    np.random.seed(0)
    bool_list = list(np.random.choice(list(bool_elements.keys()), n_rows, replace=True, p=list(bool_elements.values())))
    np.random.seed(0)
    mask = list(np.random.random(size=len(bool_list)) > 0.5)
    X_train['bool'] = [(x if m else np.nan) for (x, m) in zip(bool_list, mask)]

    # Add timestamp column
    timestamps = pd.date_range(start=pd.datetime(2019, 1, 6), end=pd.datetime(2020, 1, 20), periods=n_rows)
    timestamps = pd.Series(timestamps.values).sample(frac=1, random_state=0).values
    X_train['timestamp'] = timestamps
    
    # Non-consecutive index for extra test (simulate result after sklearn X_train-test split)
    np.random.seed(0)
    X_train.index = np.random.choice(100, X_train.shape[0], replace=False)
    
    return X_train

_generate_X_train_1()

def _generate_X_test_1():

    # Generate X_test dataframe with missing values

    # Initialise parameters
    n_rows = 10
    nominal_categories = {'stanage': 0.2, 'burbage': 0.2, 'almscliff': 0.2, 'wen_zawn': 0.2, 'hoy':0.2}

    # Generate dataframe
    np.random.seed(1)
    X_test = pd.DataFrame({
        'numeric_1': np.random.choice(100, n_rows), 
        'numeric_2': np.random.choice(10, n_rows, replace=False), 
        'numeric_3': np.random.choice(22, n_rows, replace=False), 
        'numeric_4': np.random.choice(5, n_rows), 
        'nominal': np.random.choice(list(nominal_categories.keys()), n_rows, replace=True, p=list(nominal_categories.values())), 
    })
    np.random.seed(1)
    X_test = X_test.mask((np.random.random(size=X_test.shape) > 0.6))

    # Add boolean (this doesn't work before mask applied as this turns bool dtype to float if NaNs exist)
    bool_elements = {True: 0.6, False: 0.4}
    np.random.seed(1)
    bool_list = list(np.random.choice(list(bool_elements.keys()), n_rows, replace=True, p=list(bool_elements.values())))
    np.random.seed(1)
    mask = list(np.random.random(size=len(bool_list)) > 0.25)
    X_test['bool'] = [(x if m else np.nan) for (x, m) in zip(bool_list, mask)]

    # Add timestamp column
    timestamps = pd.date_range(start=pd.datetime(2015, 1, 6), end=pd.datetime(2020, 1, 30), periods=n_rows)
    timestamps = pd.Series(timestamps.values).sample(frac=1, random_state=1).values
    X_test['timestamp'] = timestamps
    
    # Non-consecutive index for extra test (simulate result after sklearn X_test-test split)
    np.random.seed(1)
    X_test.index = np.random.choice(100, X_test.shape[0], replace=False)
    
    return X_test

_generate_X_test_1()

def test_data_1():
    return {
        'X_train': _generate_X_train_1(),
        'X_test': _generate_X_test_1(),
    }