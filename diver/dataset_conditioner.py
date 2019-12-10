# coding: utf-8

# Import libraries
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler    # Demean and rescale numerical columns
# For saving and loading FullEncoder object
import pickle
from joblib import dump, load
# For checking for nans, used in boolean_mapper
from math import isnan
from _shared import _get_boolean_elements


#############
## PRIVATE ##
#############

# Define individual encoding functions


def _missing_value_conditioner(features, useful_cols, mode='fit_transform', **kwargs):
    '''
    Function looks for missing values (NaNs) in the dataframe `features`, and deals with them as specified in the lookup table `useful_cols`.
    
    At the moment, methods for dealing with missing values are:
    - (Numeric dtype)
        - 'mean': fill with the mean value of the column
        - 'zeros': fill with zeros
    - (Nominal dtype)
        - TO DO
    - (All types)
        - 'skip': ignore this column for the missing values conditioner
        
    Parameters
    ----------
    features : pandas.DataFrame
        Main data science project input dataframe, which could include some missing values
    useful_cols : pandas.DataFrame
        Look-up dataframe, which contains information about the dtypes of desired features, and how to deal with missing values for each feature
    mode : str
        One of {'fit_transform', 'transform'}. Default = 'fit_transform. If 'transform', the feature means generated from the 'fit_transform' stage must be included as a kwarg
    **kwargs
        {'means': means} If mode is 'transform'
    
    Raises
    ------
    ValueError
        ValueError when set of 'fillna' methods in `useful_cols` contains a method not recognised
    ValueError
        ValueError when `mode` argument not recognised
    
    Returns
    -------
    features : pandas.DataFrame
        Input dataframe with missing values filled as specified
    means : pandas.Series
        Series of {feature: feature_mean} for all features specified to be filled with the feature mean, as calculated at the 'fit_transform' stage
    
    '''
    
    ## Get DataFrame containing features with some NaNs, and how to deal with them joined on from the `useful_cols` df
    # Get counts of NaN values in each column
    features_nan_counts = pd.DataFrame(features.isna().sum()).reset_index()
    # Rename columns in resulting DataFrame
    features_nan_counts.rename(mapper={'index': 'feature', 0: 'nan_count'}, axis=1, inplace=True)
    # Join on how to deal with nans from the `useful_cols` df
    features_nan_counts = pd.merge(features_nan_counts, useful_cols, on='feature')
    
    # Get set of fill types
    fill_types = set(features_nan_counts['fillna'].values)
    
    for fill_type in fill_types:
        if fill_type == 'skip':
            pass
        
        elif fill_type == 'mean':
            # Get list of columns to be filled in a given way
            feats_mean_filled = list(
                features_nan_counts['feature'][features_nan_counts['fillna'] == 'mean'].values
            )
            # Get slice of main `features` df for these features
            features_with_nans = features[feats_mean_filled]
            # Whether to calculate feature means, or use pre-calculated ones, is determined by the `mode` input parameter
            if mode == 'fit_transform':
                # Get the means of each feature
                means = features_with_nans.mean()
            elif mode == 'transform':
                # Use pre-calculated means from a previous `fit_transform` stage
                means = kwargs['means']
            else:
                raise ValueError(
                    '''`Mode` argument "{}" not supported. Needs to be one of ('fit_transform', 'transform')
                    '''.format(mode).replace('\n',' ')
                )
            # Get the locations of NaNs
            idx_nans = features_with_nans.isna()
            # Initialise copy dataframe
            features_without_nans = pd.DataFrame(index=features_with_nans.index)
            # Iterate over columns and replace NaNs for each with the column average
            for name, column in features_with_nans.iteritems():
                # Change NaNs to the mean value. This works in-place in the loop and changes features_with_nans 
                # in-place
                column.loc[idx_nans[name]] = means[name]
            # Overwrite the original features (with nans), with the new copy with nans filled
            features.loc[:,feats_mean_filled] = features_with_nans
        
        elif fill_type == 'zeros':
            # Get list of columns to be filled in a given way
            feats_zeros_filled = list(features_nan_counts['feature'][features_nan_counts['fillna'] == 'zeros'].values)
            # Get slice of main `features` df for these features
            features_with_nans_zeros = features[feats_zeros_filled]
            # Fill all NaN locations with zeros
            features_with_nans_zeros.fillna(value=0, inplace=True)
            # Overwrite the original features (with nans), with the new copy with nans filled
            features.loc[:,feats_zeros_filled] = features_with_nans_zeros
            
        else:
            raise ValueError(
                '''Currently, `_missing_value_conditioner` is programmed to deal with only dtype 'numeric' and only
                with fill methods 'mean' and 'zeros'. The full set of fill types specified in input `useful_cols`
                is {}
                '''.format(fill_types).replace('\n',' ')
            )

        # If no columns specified as to be filled by mode='means', there will currently be no means parameter to return
        # This line catches this eventuality and assigns a dummy value
        try:
            means
        except NameError:
            means = None
    
    return features, means


def _numeric_encoder(features, useful_cols, mode='fit_transform', **kwargs):
    '''
    Function standardises features by removing the mean and scaling to unit variance, using the SkLearn Standard Scaler
        
    Parameters
    ----------
    features : pandas.DataFrame
        Main data science project input dataframe, which could include some missing values
    useful_cols : pandas.DataFrame
        Look-up dataframe, which contains information about the dtypes of desired features, and how to deal with missing values for each feature
    mode : str
        One of {'fit_transform', 'transform'}. Default = 'fit_transform. If 'transform', the StandardScaler object generated from the 'fit_transform' stage must be included as a kwarg
    **kwargs
        {'scaler': scaler} If mode is 'transform'
    
    Raises
    ------
    ValueError
        ValueError when `mode` argument not recognised    
    
    Returns
    -------
    numeric_features_transformed : pandas.DataFrame
        Dataframe of transformed features
    scaler : SkLearn StandardScaler object
        StandardScaler object storing the fitted means and variances, for use when encoding test sets
        
    '''
    
    # Get list of numeric columns
    numeric_cols = useful_cols.loc[useful_cols.dtype == 'numeric', 'feature'].values
    # Select these columns from the main features dataframe
    numeric_features = features[numeric_cols]
    
    if mode == 'fit_transform':
        # Instantiate sklearn demean/rescaler
        scaler = StandardScaler()
        # Fit sklearn demean/rescaler
        numeric_features_transformed = scaler.fit_transform(numeric_features)
        
    elif mode == 'transform':
        # Load scaler included in the kwargs (already fitted in the previous 'fit_transform' stage)
        scaler = kwargs['scaler']
        # Transform numeric features using previous sklearn demean/rescaler
        numeric_features_transformed = scaler.transform(numeric_features)    
        
    else:
        raise ValueError(
            '''`Mode` argument "{}" not supported. Needs to be one of ('fit_transform', 'transform')
            '''.format(mode).replace('\n',' ')
        )

    # sklearn outputs NumPy array - reconvert to pandas DataFrame
    numeric_features_transformed = pd.DataFrame(
        numeric_features_transformed, 
        index=numeric_features.index,
        columns=numeric_features.columns,
    )
    
    return numeric_features_transformed, scaler


def _numeric_selector(features, useful_cols):
    '''
    Function skips numeric encoding and just selects the numeric features as specified in useful_cols
        
    Parameters
    ----------
    features : pandas.DataFrame
        Main data science project input dataframe, which could include some missing values
    useful_cols : pandas.DataFrame
        Look-up dataframe, which contains information about the dtypes of desired features, and how to deal with missing values for each feature
    
    Raises
    ------
    
    Returns
    -------
    numeric_features_transformed : pandas.DataFrame
        Dataframe of transformed features
        
    '''
    
    # Get list of numeric columns
    numeric_cols = useful_cols.loc[useful_cols.dtype == 'numeric', 'feature'].values
    
    # Select these columns from the main features dataframe
    numeric_features = features[numeric_cols]

    return numeric_features


def _nominal_encoder(features, useful_cols, mode='fit_transform', **kwargs):
    '''
    Function one-hot-encodes nominal features
        
    Parameters
    ----------
    features : pandas.DataFrame
        Main data science project input dataframe, which could include some missing values
    useful_cols : pandas.DataFrame
        Look-up dataframe, which contains information about the dtypes of desired features, and how to deal with
        missing values for each feature
    mode : str
        One of {'fit_transform', 'transform'}. Default = 'fit_transform. If 'transform', the function will calculate the set difference between train and test one-hot-encoded categories (train set = `cat_cols`). 
            - Train set categories missing in the test set will be added on as columns of zeros
            - Test set categories missing in the train set will be dropped
            - Test set categories will finally be sorted in the same order as train set
    **kwargs
        {'cat_cols': cat_cols} If mode is 'transform'
    
    Raises
    ------
    ValueError
        ValueError when `mode` argument not recognised     
    
    Returns
    -------
    nominal_features_transformed : pandas.DataFrame
        Dataframe of transformed features
    cat_cols : list
        List of the categorical columns to ensure future encoded test sets contain the same ordered list of 
        categories
    '''
    
    # Get list of nominal columns
    nominal_cols = useful_cols.loc[useful_cols.dtype == 'nominal', 'feature'].values

    # Select these columns from the main features dataframe
    nominal_features = features[nominal_cols]
    
    # If there are no nominal columns, skip the later steps
    if len(nominal_cols) == 0:
        nominal_features_transformed = nominal_features
        cat_cols = []
        
    # If there are some nominal columns:    
    elif len(nominal_cols) > 0:


        # Use pandas `get_dummies` method to one-hot-encode these columns
        nominal_features_transformed = pd.get_dummies(nominal_features)

        if mode == 'fit_transform':

            # Store transformed one-hot category columns for use later with the test set
            cat_cols = list(nominal_features_transformed.columns)

        elif mode == 'transform':

            # Load 'fit_transform' stage categorical columns
            cat_cols = kwargs['cat_cols']

            # Test set columns
            test_cat_cols = list(nominal_features_transformed.columns)

            # Train set categories missing in the test set
            missing_test_cols = set(cat_cols) - set(test_cat_cols)

            print('missing_test_cols: {}'.format(missing_test_cols))

            # Test set categories missing in the train set
            extra_test_cols = set(test_cat_cols) - set(cat_cols)

            print('extra_test_cols: {}'.format(extra_test_cols))

            # Remove extra cols
            nominal_features_transformed.drop(extra_test_cols, axis=1, inplace=True)

            # Add in missing cols as all zeros
            for col in missing_test_cols:
                nominal_features_transformed[col] = 0


            print('set difference after sorting: {}'.format(set(cat_cols) - set(nominal_features_transformed.columns)))

            # Sort columns in same order as train set
            nominal_features_transformed = nominal_features_transformed[cat_cols]

            # Check columns are now identical between train and test sets
            print('Test set cols in same order as train set: {}'.format(
                cat_cols == list(nominal_features_transformed.columns)))

        else:
            raise ValueError(
                '''`Mode` argument "{}" not supported. Needs to be one of ('fit_transform', 'transform')
                '''.format(mode).replace('\n',' ')
            )
        
    return nominal_features_transformed, cat_cols


def _weekend_flagger(timestamp):
    '''Given a timestamp, returns 1 if it is a Saturday or Sunday, 0 otherwise'''
    return timestamp.weekday() == 6 or timestamp.weekday() == 7


def _timestamp_transformer(timestamps, time_of_day_in='seconds', year_normalised=True):
    
    '''
    Function which takes in a Pandas Series of timestamps and returns useful features derived from the timestamp:
    - Time of day 
    - Day of week
    - Month of year
    - Year
    - Flag if it is a weekend (boolean)

    All but the year column are cyclical, so are further decomposed into sin and cos transforms of the original, \
    so that e.g. 11.59pm is considered close to 00.00am, and Sunday and Monday, and December and January \
    are considered close together

    Parameters
    ----------
    timestamps : pandas.Series
        Series of timestamp data
    time_of_day_in : str
        One of ['seconds', 'hours'] - specifies whether time of day is computed in hours (24) or seconds (24*60*60)
    year_normalised : bool
        Specifies whether the non-cyclical `year` column should be demeaned and rescaled or not

    Returns
    -------
    timestamps_transformed : pandas.DataFrame
        df of the above encodings of the timestamps Series passed in
        
    '''
    
    # Ensure input timestamps are in timestamp/datetime format and not string format
    timestamps = pd.to_datetime(timestamps)
    
    # Split timestamps in timestamps series into a pandas DataFrame of component timestamp parts
    timestamps_transformed = timestamps.apply(
        lambda x: {
            'day_of_week': x.weekday(),
            'day_of_month': x.day, 
            'month_of_year': x.month, 
            'year': x.year, 
            'hour_of_day': x.hour, 
            'minute_of_hour': x.minute, 
            'second_of_minute': x.second,
            'is_weekend': _weekend_flagger(x)}
    )

    timestamps_transformed = pd.DataFrame(list(timestamps_transformed))

    # Get second of day 
    timestamps_transformed['second_of_day'] = timestamps_transformed['hour_of_day']*60*60 + timestamps_transformed['minute_of_hour']*60 + timestamps_transformed['second_of_minute']

    # Define constants
    seconds_in_day = 24*60*60
    weekdays_in_week = 7
    months_in_year = 12
    hours_in_day = 24

    # Circular transform of second of day
    timestamps_transformed['sin_second_of_day'] = timestamps_transformed['second_of_day'].apply(lambda x: np.sin(2*np.pi*x / seconds_in_day))
    timestamps_transformed['cos_second_of_day'] = timestamps_transformed['second_of_day'].apply(lambda x: np.cos(2*np.pi*x / seconds_in_day))

    # Circular transform of hour of day
    timestamps_transformed['sin_hour_of_day'] = timestamps_transformed['hour_of_day'].apply(lambda x: np.sin(2*np.pi*x / hours_in_day))
    timestamps_transformed['cos_hour_of_day'] =  timestamps_transformed['hour_of_day'].apply(lambda x: np.cos(2*np.pi*x / hours_in_day))

    # Circular transform of day of week
    timestamps_transformed['sin_day_of_week'] = timestamps_transformed['day_of_week'].apply(lambda x: np.sin(2*np.pi*x / weekdays_in_week))
    timestamps_transformed['cos_day_of_week'] = timestamps_transformed['day_of_week'].apply(lambda x: np.cos(2*np.pi*x / weekdays_in_week))

    # Circular transform of month of year
    timestamps_transformed['sin_month_of_year'] = timestamps_transformed['month_of_year'].apply(lambda x: np.sin(2*np.pi*x / months_in_year))
    timestamps_transformed['cos_month_of_year'] = timestamps_transformed['month_of_year'].apply(lambda x: np.cos(2*np.pi*x / months_in_year))

    # Determine list of output columns based on `time_of_day_in` parameter
    if time_of_day_in == 'seconds':
        output_cols = [
            'sin_second_of_day',
            'cos_second_of_day',
            'sin_day_of_week',
            'cos_day_of_week',
            'sin_month_of_year',
            'cos_month_of_year',
            'year',
            'is_weekend',
        ]
    elif time_of_day_in == 'hours':
            output_cols = [
            'sin_hour_of_day',
            'cos_hour_of_day',
            'sin_day_of_week',
            'cos_day_of_week',
            'sin_month_of_year',
            'cos_month_of_year',
            'year',
            'is_weekend',
        ]
    else:
        raise ValueError('`time_of_day_in` should be one of [\'seconds\', \'hours\']')

    # If specified that the `year` column should be normalised (default), use SkLearn 
    # demean and rescaling on this column
    if year_normalised == True:

        # Instantiate sklearn standard scaler (demeans and rescales)
        scaler = StandardScaler()

        # Fit sklearn standard scaler
        rescaled_years = scaler.fit_transform(timestamps_transformed['year'].values.reshape(-1, 1))

        # Update `year` column with rescaled version
        timestamps_transformed['year'] = rescaled_years

    elif year_normalised != False:
        raise ValueError('`year_normalised` should be boolean - `True` for normalising the `year` column,         and `False` otherwise')

    # Keep only desired output columns
    timestamps_transformed = timestamps_transformed[output_cols]
    
    # Append original Series name to all column names of dataframe 
    # (so that, in the case of multiple timestamps Series being transformed, they can be distinguished and 
    # concatenated)
    
    # Get name of timestamps Series
    name = str(timestamps.name)
    
    # Append original Series name to all column strings
    renamed_cols = []
    columns = timestamps_transformed.columns
    for column in columns:
        renamed_cols.append(name + '_' + str(column))   
    
    # Get the dictionary for renaming the columns
    mapper = dict(zip(columns, renamed_cols))
    
    # Rename the output DataFrame columns
    timestamps_transformed = timestamps_transformed.rename(mapper=mapper, axis='columns')

    return timestamps_transformed


def _timestamp_encoder(features, useful_cols):
    '''
    Function one-hot-encodes timestamp features
        
    Parameters
    ----------
    features : pandas.DataFrame
        Main data science project input dataframe, which could include some missing values
    useful_cols : pandas.DataFrame
        Look-up dataframe, which contains information about the dtypes of desired features, and how to deal with missing values for each feature
    
    Raises
    ------
    
    
    Returns
    -------
    timestamp_features_transformed : pandas.DataFrame
        Dataframe of transformed features

    '''
    
    # Get list of timestamp columns
    timestamp_cols = useful_cols.loc[useful_cols.dtype == 'timestamp', 'feature'].values

    # Select these columns from the main features dataframe
    timestamp_features = features[timestamp_cols]

    # Initialise empty pandas Dataframe, for appending to in the loop
    timestamp_features_transformed = pd.DataFrame(index=timestamp_features.index)
    # Loop over all timestamp columns
    for column in timestamp_features:
        # Derive features from timestamp
        column_transformed = _timestamp_transformer(timestamp_features[column])
        # Append to final timestamp dataframe
        timestamp_features_transformed = pd.concat([timestamp_features_transformed, column_transformed], axis=1)
        
    return timestamp_features_transformed


def _boolean_mapper(boolean_like):
    '''
    Function maps boolean-like inputs (str or numeric) to true bool
    '''
    
    # Inputs to be recognised as booleans
    true_set, false_set = _get_boolean_elements()
    
    # Test for nans, and if nan, return itself (nans dealt with later)
    if type(boolean_like) in {float, int} and isnan(boolean_like):
        return boolean_like
    # Convert to true boolean values
    elif boolean_like in true_set:
        return True
    elif boolean_like in false_set:
        return False
    else:
        raise ValueError(f'Boolean-like input {boolean_like} (dtype {type(boolean_like)}), not recognised as a boolean')

        
def _boolean_transformer(features, useful_cols):
    '''
    Function which takes in a Pandas DataFrame of columns marked as boolean but which may be str or numeric, and ensures all values are true booleans

    Parameters
    ----------
    features : pandas.DataFrame
        Input dataframe
    useful_cols : pandas.DataFrame
        Look-up dataframe, which contains information about the dtypes of desired features, and how to deal with missing values for each feature

    Returns
    -------
    bool_features_transformed : pandas.DataFrame
        A DataFrame with any string representations replaced with true booleans
    
    '''

    # Map boolean-like features to true booleans
    bool_cols = useful_cols.loc[useful_cols.dtype == 'bool', 'feature'].values
    bool_features_transformed = features[bool_cols].applymap(_boolean_mapper)

    return bool_features_transformed
        

############
## PUBLIC ##
############

# Define global encoding functions

class FullEncoder:
    '''
    The FullEncoder object performs the following data conditioning and encoding actions:
            - Fills in missing values as specified in the lookup table `useful_cols`
            - Encodes numeric features using the SkLearn StandardScaler (default)
            - Encodes nominal features using pandas.get_dummies
            - Encodes timestamp features - various cyclical features generated from timestamp data
            - Ensures features specified as boolean are boolean (can sometimes be read in as strings or ints)
    
    FullEncoder contains the following SkLearn-format methods
        - fit_transform: 'fits' the encodings (means variances, categories) which are stored as instance attributes to be reused on later datasets, then transforms the data
        - transform: transforms a dataset without fitting (uses previously learnt encoding attributes)

    Parameters
    ----------

    Attributes
    ----------

    means_ : pandas.Series
        Series of {feature: feature_mean} for all features specified to be filled with the feature mean, as calculated at the 'fit_transform' stage
    scaler_ : SkLearn StandardScaler object
        StandardScaler object storing the fitted means and variances, for use when encoding test sets
    cat_cols_ : list
        List of the categorical columns to ensure future encoded test sets contain the same ordered list of categories
    
    '''

    # Initialiser / instance attributes
    def __init__(self):

        # Attributes will be generated when `fit_transform` is called and used when `transform` is called
        self.means_ = None
        self.scaler_ = None
        self.cat_cols_ = None


    def fit_transform(self, df, useful_cols, encode_numeric=True):
        '''
        Function performs the following encoding actions:
            - Fills in missing values as specified in the lookup table `useful_cols`. Stores any column means as object attributes for later use
            - Encodes numeric features using the SkLearn StandardScaler (default), or leaves numeric columns unaltered if specified. Stores any scaler object generated for later use
            - Encodes nominal features using pandas.get_dummies. Stores list of categorical columns generated for later use
            - Encodes timestamp features - various cyclical features generated from timestamp data
            - Ensures features specified as boolean are boolean (can sometimes be read in as strings or ints)

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe
        useful_cols : pandas.DataFrame
            Look-up dataframe, which contains information about the dtypes of desired features, and how to deal with
            missing values for each feature
        encode_numeric : bool
            Boolean flag for whether to demean and scale numeric data to unit variance (default) or skip this encoding (useful for intelligibility of decision tree-type models)
            
        Raises
        ------

        Returns
        -------
        features_encoded : pandas.DataFrame
            Subset of useful features from `df`, encoded as specified in `useful_cols`
        
        '''
        
        # Get subset of useful features in `df` according to the lookup `useful_cols`
        features = df[list(useful_cols['feature'])]

        # Fill in missing values
        print('Filling in missing values...')
        features, self.means_ = _missing_value_conditioner(features, useful_cols, mode='fit_transform')
        print('Missing values filled')
        
        # Encode numeric features and store the fitted StandardScaler object for use with the test set
        if encode_numeric == True:
            print('Encoding numeric features...')
            numeric_features_transformed, self.scaler_ = _numeric_encoder(features, useful_cols, mode='fit_transform')
            print('Numeric features encoded')

        elif encode_numeric == False:
            print('Numeric features passing through without encoding...')
            numeric_features_transformed = _numeric_selector(features, useful_cols)
            self.scaler_ = 'no_scaler'
            print('Numeric features passed through without encoding')
        
        # Encode nominal features and store a list of the resulting categorical columns for use with the test set
        print('Encoding nominal features...')
        nominal_features_transformed, self.cat_cols_ = _nominal_encoder(features, useful_cols, mode='fit_transform')
        print('Nominal features encoded')
        
        # Encode timestamp features
        print('Encoding timestamp features...')
        timestamp_features_transformed = _timestamp_encoder(features, useful_cols)
        print('Timestamp features encoded')
        
        # Encode boolean features (some are str or int when loaded via pandas.read_csv)
        print('Encoding boolean features...')
        bool_features_transformed = _boolean_transformer(features, useful_cols)
        print('Boolean features encoded')

        # Concatenate all constituent dfs into final df 
        features_encoded = pd.concat(
            [
                numeric_features_transformed, 
                nominal_features_transformed, 
                timestamp_features_transformed,
                bool_features_transformed,
            ],
            axis=1,
        )

        return features_encoded


    def transform(self, df, useful_cols):

        '''
        Function performs the following encoding actions, using attributes (means, variances, categorical columns) as generated at the `fit_transform` stage:
            - Fills in missing values as specified in the lookup table `useful_cols`. Attributes as generated at the `fit_transform` stage
            - Encodes numeric features using the SkLearn StandardScaler (default), or leaves numeric columns unaltered if specified. Attributes as generated at the `fit_transform` stage
            - Encodes nominal features using pandas.get_dummies. Attributes as generated at the `fit_transform` stage
            - Encodes timestamp features - various cyclical features generated from timestamp data
            - Ensures features specified as boolean are boolean (can sometimes be read in as strings or ints)

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe
        useful_cols : pandas.DataFrame
            Look-up dataframe, which contains information about the dtypes of desired features, and how to deal with
            missing values for each feature
            
        Raises
        ------

        Returns
        -------
        features_encoded : pandas.DataFrame
            Subset of useful features from `df`, encoded as specified in `useful_cols`
        
        '''
        
        # Get subset of useful features in `df` according to the lookup `useful_cols`
        features = df[list(useful_cols['feature'])]
        
        # Fill in missing values
        print('Filling in missing values...')
        features, self.means_ = _missing_value_conditioner(
            features, 
            useful_cols, 
            mode='transform', 
            means=self.means_
        )
        print('Missing values filled')
        
        # Whether numeric features are demeaned/rescaled or left as they are depends on whether the `self.scaler_` attribute is a StandardScaler object or a string ('no_scaler')
        if type(self.scaler_) == StandardScaler:

            # Encode numeric features and store the fitted StandardScaler object for use with the test set
            print('Encoding numeric features...')
            numeric_features_transformed, self.scaler_ = _numeric_encoder(
                features, 
                useful_cols, 
                mode='transform', 
                scaler=self.scaler_
            )
            print('Numeric features encoded')

        elif type(self.scaler_) == str:

            # Encode numeric features and store the fitted StandardScaler object for use with the test set
            print('Numeric features passing through without encoding...')
            numeric_features_transformed = _numeric_selector(features, useful_cols)
            print('Numeric features passed through without encoding')

        else:
            raise ValueError('`scaler_` instance attribute must be of a type either (StandardScaler, str) but is of type {}. Make sure `fit_transform` has been performed before using `transform`'.format(type(self.scaler_)))
        
        # Encode nominal features and store a list of the resulting categorical columns for use with the test set
        print('Encoding nominal features...')
        nominal_features_transformed, self.cat_cols_ = _nominal_encoder(
            features, 
            useful_cols, 
            mode='transform', 
            cat_cols=self.cat_cols_
        )
        print('Nominal features encoded')
        
        # Encode timestamp features
        print('Encoding timestamp features...')
        timestamp_features_transformed = _timestamp_encoder(features, useful_cols)
        print('Timestamp features encoded')
        
        # Encode boolean features (some are str or int when loaded via pandas.read_csv)
        print('Encoding boolean features...')
        bool_features_transformed = _boolean_transformer(features, useful_cols)
        print('Boolean features encoded')

        # Concatenate all constituent dfs into final df 
        features_encoded = pd.concat(
            [
                numeric_features_transformed, 
                nominal_features_transformed, 
                timestamp_features_transformed,
                bool_features_transformed,
            ],
            axis=1,
        )

        return features_encoded


    def save_encoder(self, pathname):
        '''
        Function saves the current FullEncoder object, using pickle

        parameters
        ----------

        pathname : str
            Desired pathname for the saved object (do not include '.pkl')

        Raises
        ------

        Returns
        -------

        '''

        with open(pathname + '.pkl', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


def load_encoder(pathname):
    '''
    Function loads a FullEncoder object saved via `save_encoder`

    parameters
    ----------

    pathname : str
        Pathname for the saved encoder (do not include '.pkl')

    Raises
    ------

    Returns
    -------

    encoder : FullEncoder object

    '''

    with open((pathname + '.pkl'), 'rb') as f:
        encoder = pickle.load(f)
        return encoder
