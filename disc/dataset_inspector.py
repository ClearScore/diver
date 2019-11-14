# Import libraries
import pandas as pd

def categorical_excess_cardinality_flagger_and_reducer(
    df, 
    useful_cols, 
    cardinality_fraction_threshold=0.1, 
    cardinality_max_categories_threshold=50,
    **kwargs
):

    '''
    Function inspects categorical features in the input dataframe where there is excess cardinality (too many unique \
    values), and returns a list of features which exceed these thresholds, as well as Series detailing the cardinality \
    (in absolute terms as well as as a fraction of the total number of samples).
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe
    useful_cols : pandas.DataFrame
        Look-up dataframe, which contains information about the dtypes of desired features, and how to deal with
        missing values for each feature
    cardinality_fraction_threshold : int or None
        Cardinality fraction is the ratio of unique values for a given feature to the total number of samples in the dataset
        - If None, no threshold applied
        - If int, this is the threshold beyond which a feature is deemed to have excessive cardinality
    cardinality_max_categories_threshold : int or None
        Cardinality max categories is the absolute number of unique values for a given feature
        - If None, no threshold applied
        - If int, this is the threshold beyond which a feature is deemed to have excessive cardinality
        
    Returns
    -------
    df_cardinality : pandas.Series
        Cardinality (number of unique elements) of the categorical features in the input dataset
    df_cardinality_fraction : pandas.Series
        Cardinality fractions of the categorical features in the input dataset
    excess_cardinality : list
        List of features which exceed one or more of the thresholds
    
    '''
    
    # Get categorical feature names
    categorical_features = list(useful_cols.loc[useful_cols.dtype == 'nominal', 'feature'])

    # Get categorical features
    categorical_data = df[categorical_features]

    # Get total number of samples
    m_samples = categorical_data.shape[0]

    # Get cardinality of categorical features
    df_cardinality = categorical_data.nunique()

    # Get cardinality as a fraction of total number of samples in the dataset 
    df_cardinality_fraction = df_cardinality / m_samples

    # Get features which exceed cardinality_fraction_threshold
    excess_cardinality_fraction = set(df_cardinality_fraction[df_cardinality_fraction > cardinality_fraction_threshold].index)
    print(
        f'Following features exceed cardiality fraction threshold of {cardinality_fraction_threshold}: {excess_cardinality_fraction}'
    )    
    
    # Get features which exceed cardinality_max_categories_threshold
    excess_cardinality_absolute = set(df_cardinality[df_cardinality > cardinality_max_categories_threshold].index)
    print(
        f'Following features exceed cardiality absolute number threshold of {cardinality_max_categories_threshold}: {excess_cardinality_absolute}'
    )
    
    # Get final list of features which exceed the thresholds
    excess_cardinality = list(set.union(excess_cardinality_fraction, excess_cardinality_absolute))
    
    if kwargs['reducer'] == True:
        
        # Get number of max categories for categorical features
        n_max_categories = kwargs['reducer_max_categories']
        
        for feature in excess_cardinality:
            # Cap categories at n_max_categories - smaller categories, assign to 'other'
            # Value counts for all categories in the categorical feature - ordered list
            value_counts = df[feature].value_counts()

            # n_max_categories sets the cap on desired total categories - get a list of smaller categories beyond this threshold
            small_categories = list(value_counts[n_max_categories:].index)

            # Assign these smaller categories to a placeholder
            df.loc[df[feature].isin(small_categories), feature] = 'SMALL_CATEGORY'
            
        print(f'Features {excess_cardinality} exceeded thresholds and have each been reduced down to a maximum {n_max_categories} categories per feature')
            
    return df_cardinality, df_cardinality_fraction, excess_cardinality


def excess_missing_data_flagger_and_cutter(df, useful_cols, nan_fraction_threshold=0.15, cutter=True):
    '''
    Function calculates `null_count_density` (number of missing values for each feature as a fraction of the /
    number of samples), and, if specified, culls these from the input dataframe (inplace)
    
    Parameters
    ----------
    df : pandas.DataFrame
        Desired features
    useful_cols : pandas.DataFrame
        Lookup table detailing desired features
    nan_fraction_threshold : int
        Threshold for fraction of samples with missing values (for a given feature) above which the feature is deemed unacceptable
    cutter : bool
        If true [default], df is modified to remove features with an unacceptably large fraction of missing values
    
    Returns
    -------
    null_counts : pandas.Series
        Record of number of missing values for each feature
    null_count_density : pandas.Series
        Record of number of missing values for each feature as fraction of number of samples
    excess_null_count : list
        List of features where the null count density exceeds `nan_fraction_threshold`
    useful_cols : pandas.DataFrame
        Returned unchanged if `cutter` == False, else the clipped features are removed
    '''

    # Get counts of nulls for each column
    null_counts = df.isna().sum()
    # Get fraction of column which is null values
    null_count_density = null_counts / df.shape[0]

    # Get list of columns which don't satisfy the nan threshold
    excess_null_count = null_count_density[null_count_density > nan_fraction_threshold]

    # Store a record of columns with unacceptably high null count
    excess_null_count = list(null_count_density[null_count_density > nan_fraction_threshold].index)

    # Filter columns with higher than acceptable null density if specified
    if (cutter == True) and (len(excess_null_count) > 0):
        print(f'Following features dropped for having a null count density greater than {nan_fraction_threshold}: {excess_null_count} and have been dropped')
        df.drop(excess_null_count, axis=1, inplace=True)
        # Drop culled features from `useful_features`
        useful_cols = useful_cols.loc[~useful_cols.feature.isin(excess_null_count), :]
        
    return null_counts, null_count_density, excess_null_count, useful_cols


def zero_variance_flagger_and_cutter(df, useful_cols, cutter=True):
    '''
    Function flags features with only one value (zero variance), and, if specified, culls these from the input dataframe (inplace)
    
    Parameters
    ----------
    df : pandas.DataFrame
        Desired features
    useful_cols : pandas.DataFrame
        Lookup table detailing desired features
    cutter : bool
        If true [default], df is modified to remove features with only one value
    
    Returns
    -------
    zero_variance_features : list
        List of features with zero variance
    useful_cols : pandas.DataFrame
        Returned unchanged if `cutter` == False, else the clipped features are removed
    '''

    # Check for columns with only one value
    zero_variance_features = []
    for feature_name, feature in df.iteritems():
        if len(feature.value_counts()) == 1:
            zero_variance_features.append(feature_name)
            
    if cutter == True:
        print(f'Following features dropped for having zero variance: {zero_variance_features}')
        df.drop(zero_variance_features, axis=1, inplace=True)
        # Drop culled features from `useful_features`
        useful_cols = useful_cols.loc[~useful_cols.feature.isin(zero_variance_features), :]
    
    return zero_variance_features, useful_cols


def inspect_and_clean(df,
                      useful_cols,
                      cardinality_args = {
                        'cardinality_fraction_threshold': 0.05,
                        'cardinality_max_categories_threshold':1000,
                        'reducer_max_categories':10
                        },
                      nan_fraction_threshold=0.15,
                      reduce_and_cull=True,
                     ):
    '''
    Performs the following checks (and corrects if specified) given the input parameter thresholds
    - Check for unacceptably high cardinality (number of categories per categorical feature) (and reduces if specified)
    - Check for unacceptably high numbers of missing values per feature (and removes these if specified)
    - Check for features with zero variance (and removes these if specified)
    
    Parameters
    ----------
    df : pandas.DataFrame
        Desired features
    useful_cols : pandas.DataFrame
        Lookup table detailing desired features
    cardinality_args : dict
        Thresholds for cardinality inspector function
    nan_fraction_threshold : float
        Threshold for missing data function
    reduce_and_cull : bool
        If True, all inspection functions ameliorate `df` by removing offending features or reducing their cardinality
    
    Returns
    -------
    useful_cols : pandas.DataFrame
        Returned unchanged if `cutter` == False, else the clipped features are removed
    report : dict
        Report of all the outputs of individual inspection functions - cardinality and missing data counts lists of offending features for all inspection stages 
    '''
    
    # Filter just the useful columns
    df_altered = df.loc[:, list(useful_cols.feature)]

    # Check for excess cardinality and reduce cardinality of offending features, if specified
    df_cardinality, df_cardinality_fraction, excess_cardinality = categorical_excess_cardinality_flagger_and_reducer(
        df_altered, 
        useful_cols, 
        cardinality_fraction_threshold=cardinality_args['cardinality_fraction_threshold'], 
        cardinality_max_categories_threshold=cardinality_args['cardinality_max_categories_threshold'],
        reducer=reduce_and_cull,
        reducer_max_categories=cardinality_args['reducer_max_categories']
    )

    # Check for excess missing values and remove offending features if specified
    null_counts, null_count_density, excess_null_count, useful_cols = excess_missing_data_flagger_and_cutter(
        df_altered,
        useful_cols,
        nan_fraction_threshold=nan_fraction_threshold,
        cutter=reduce_and_cull
    )

    # Check for features with zero variance (only one value) and remove offending features if specified
    zero_variance_features, useful_cols = zero_variance_flagger_and_cutter(
        df_altered,
        useful_cols,
        cutter=reduce_and_cull
    )

    # Compile record
    report = {}
    report['cardinality_pre'] = df_cardinality
    report['cardinality_fraction_pre'] = df_cardinality_fraction
    report['excess_cardinality_features'] = excess_cardinality
    report['null_count_pre'] = null_counts
    report['null_fraction_pre'] = null_count_density
    report['excess_nulls_features'] = excess_null_count
    report['zero_variance_features'] = zero_variance_features
    
    return useful_cols, report