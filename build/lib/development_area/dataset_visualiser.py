import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def correlation_plotter(corr_plot):

    fig = plt.figure(figsize=[15, 12])

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr_plot, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Generate a custom diverging colormap
    cmap = sns.color_palette("RdBu_r", 9)

    ax = sns.heatmap(corr_plot, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=True, fmt='.2f')

    
def correlation_analyser(df_encoded, label, n_top_correlating_features=10, plot=True):
    
    # Create a copy of the input dataframe
    df_corr = df_encoded.copy()

    # Create a full dataset of encoded features alongside the dependent variable (label)
    df_corr['dependent_variable'] = label

    # Get correlation matrix
    corr = df_corr.corr()

    # Get absolute correlation strength for sorting (care equally about positive and negative correlations to the label)
    corr['abs_dependent_variable'] = corr['dependent_variable'].apply(lambda x: np.abs(x))

    # Get top n correlating features (sorted by abs of the correlation coefficient)
    top_n_correlates = corr.loc[corr.abs_dependent_variable.sort_values(ascending=False).drop(['dependent_variable']).index, 'dependent_variable'].head(n_top_correlating_features)

    if plot == True:
    
        # Get list of top n correlating features by abs strength
        top_n_correlate_names = list(top_n_correlates.index)
        top_n_correlate_names.append('dependent_variable')

        # Get data for plotting
        corr_plot = corr.loc[top_n_correlate_names, top_n_correlate_names]

        # Plot
        correlation_plotter(corr_plot)
        
    return top_n_correlates