####################
# Import libraries #
####################

# standard numerical libraries
import numpy as np
import pandas as pd

# Plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Sklearn model evaluation
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

# For generating multi-colour confusion matrices
from numpy.ma import masked_array

import itertools


########################
# Individual functions #
########################

def evaluate_classifiers(trained_classifiers, X, y_true):
    '''
    Calculates predictions and classifier statistics from a dictionary of trained sklearn classifiers
    
    Parameters
    ----------
    trained_classifiers : dict of sklearn classifiers
    X : pandas.DataFrame
        Feature set
    y_true : pandas.Series
        Corresponding label set
        
    Returns
    -------
    y_scores : pandas.DataFrame
        True labels ('LABELS') vs probability scores for all classifiers
    y_preds : pandas.DataFrame
        True labels ('LABELS') vs predicted classes for all classifiers
    classifier_metrics : pandas.DataFrame
        Summary statistics for classifier performance
    roc_curves : dict
        Dictionary of arrays for plotting ROC curves
    confusion_matrices : dict
        Dictionary of arrays for plotting confusion matrices
    '''
    
    y_preds = {'LABELS': y_true.values}
    y_scores = {'LABELS': y_true.values}

    accuracies = {}
    f1_scores = {}
    roc_auc_scores = {}
    gini_scores = {}

    roc_curves = {}
    confusion_matrices = {}

    for name, clf in trained_classifiers.items():

        # Store the classic accuracy score
        accuracies[name] = clf.score(X, y_true)

        # Calculate the F1 scores
        y_pred = clf.predict(X)
        f1_scores[name] = f1_score(y_true, y_pred)
        y_preds[name] = y_pred

        # Calculate and store ROC curves and AUC scores
        y_score = clf.predict_proba(X)[:, 1]
        y_scores[name] = y_score
        roc_curves[name] = roc_curve(y_true, y_score)
        roc_auc_scores[name] = roc_auc_score(y_true, y_score)
        gini_scores[name] = 2*roc_auc_scores[name] - 1

        # Store confusion matrices
        confusion_matrices[name] = confusion_matrix(y_true, y_pred)

    # Compile results DataFrames
    y_scores = pd.DataFrame(y_scores)
    y_preds = pd.DataFrame(y_preds)

    classifier_metrics = pd.DataFrame({
        'accuracy': accuracies,
        'f1_score': f1_scores, 
        'roc-auc': roc_auc_scores,
        'gini': gini_scores
    })

    return y_scores, y_preds, classifier_metrics, roc_curves, confusion_matrices


def plot_roc(roc_curves, baseline=True, perfect_clf_line=True, color_palette='standard'):
    
    '''
    Plots ROC curves
    
    Parameters
    ----------
    roc_curves : dict
        Dict of {'model_name': sklearn ROC parameters}
    baseline : bool
        If True, plots diagonal random-guesser line
    perfect_clf_line : bool
        If True, plots perfect classifier line
    color_palette : str
        One of {'standard', 'comparison'}:
            - If 'standard', uses standard seaborn categorical color_palette
            - If 'comparison', plots one line blue, and one line red - useful for comparing new and old models back to back (has to be 2 models in `roc_curves` only)
    '''
    
    plt.figure(figsize=[20, 10])

    # Plot baselines if specified
    if baseline:
        baseline = plt.plot((0, 1), (0, 1), 'k--', label='baseline')
    if perfect_clf_line:
        perfect = plt.plot((0, 0, 1), (0, 1, 1), '--', color='#FF33F0', label='perfect_classifier')

    # Select plot palette
    if color_palette == 'standard':
        colours = sns.color_palette(n_colors=len(roc_curves))
    elif color_palette == 'comparison':
        if len(roc_curves) == 2:
            colours = ['b', 'r']
        else:
            raise ValueError('Input only 2 roc curves for "comparison" color_palette') 

    # Plot all ROC curves
    for plot_line_number, (model_name, roc) in enumerate(roc_curves.items()):
        fpr, tpr, thresholds = roc
        plt.plot(fpr, tpr, '-', color=colours[plot_line_number], label=model_name)

    plt.title('ROC Curve', fontsize='xx-large')
    plt.xlabel('False Positive Rate \n (False Positives / All Negatives)', fontsize='x-large')
    plt.ylabel('True Positive Rate \n (True Positives / All Positives)', fontsize='x-large')

    plt.legend(fontsize='x-large');
    
    
def generate_label_palettes(categorical_palette, n_labels=2, plotter='mpl'):
    '''
    Given a parent seaborn categorical palette, generates single-colour sequential palettes for each colour in the parent palette, up to n_labels colours
    
    Parameters
    ----------
    categorical_palette : seaborn.palettes._ColorPalette
        Parent palette of various colours
    n_labels : int
        Number of labels (dependent variables) in the classification task
    plotter : str
        One of {'sns', 'mpl'}:
            - 'sns' dictates the output palettes will be in `seaborn` color_palette format
            - 'mpl' dictates the output palettes will be in `matplotlib` colormap format
            
    Returns
    -------
    label_palettes : dict
        Dictionary of format {'label_name': single-colour map/palette}
    '''

    label_palettes = {}
    for i in range(n_labels): 
        if plotter == 'sns':
            label_palettes[f'label_{i}'] = sns.light_palette(categorical_palette[i], n_colors=50)
        elif plotter == 'mpl':
            label_palettes[f'label_{i}'] = ListedColormap(sns.light_palette(categorical_palette[i], n_colors=50).as_hex())
        else:
            raise ValueError(f'plotter type {plotter} not recognised')

    return label_palettes


def plot_confusion_matrix(cm, classes, title='Confusion Matrix', fig=None, index=111, categorical_palette=sns.color_palette()):
    '''
    Plots confusion matrix, with class colours consistent with other plots
    
    Parameters
    ----------
    cm : np.array
        Confusion matrix array
    classes : int
        Number of unique classes in the target variable
    normalize : bool
        Whether to display absolute or proportional values in the confusion matrix
    title : str
        Title of plot
    '''
    
#     fig, ax = plt.subplots(figsize=[5, 5])
    if fig is None:
        fig = plt.figure(figsize=[5, 5])
    ax = fig.add_subplot(index)

    # Normalise confusion matrix, for color scale and, optionally, for text display
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Generate varying palettes for each class from parent categorical palette
    label_palettes = generate_label_palettes(categorical_palette, n_labels=cm.shape[0], plotter='mpl')

    for i, label in enumerate(cm_norm):

        # Mask confusion matrix for each label, in order to apply a separate label colormap for each
        mask = np.zeros_like(cm_norm)
        # Imshow builds from bottom to top; row index for confusion matrix array is the opposite
        inverted_index = mask.shape[0] - 1 - i
        mask[inverted_index, :] = 1
        cm_masked = masked_array(cm_norm, mask)

        # Get label color palette
        cmap = label_palettes[f'label_{i}']

        # Plot label color intensities, based on normalised values
        cm_label = ax.imshow(cm_masked, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
        
    # Plot confusion matrix values
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        text = str(round(cm[i, j], 2)) + '\n' + str(round(100*cm_norm[i, j], 1)) + '%'
        ax.text(j, i+0.07, text,
                 horizontalalignment="center",
                 color="white" if cm_norm[i, j] > 0.5 else "black",
                 fontsize='x-large')

    # Formatting
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize='x-large')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize='x-large')
    
#     ax.tight_layout()
    ax.set_title(title, fontsize='xx-large')
    ax.set_xlabel('Predicted Label', fontsize='x-large')
    ax.set_ylabel('True Label', fontsize='x-large')

#     plt.show()
    return ax


def get_subplot_dims(num_plots, max_cols=3):
    '''Return subplot layout for a given number of plots'''
    if num_plots <= max_cols:
        num_rows = 1
        num_cols = num_plots
    else:
        num_rows = num_plots // max_cols
        remainder = num_plots % max_cols
        if remainder > 0:
            num_rows += 1
        num_cols = max_cols
    return num_rows, num_cols


def plot_confusion_matrices(confusion_matrices, y_preds):
    '''
    Plots a series of confusion matrices from a dictionary of CM arrays, from a set of trained models
    
    Parameters
    ----------
    confusion_matrices : dict
        Dict of confusion matrix arrays
    '''
    
    
    num_rows, num_cols = get_subplot_dims(len(confusion_matrices))

    fig, ax = plt.subplots(num_rows, num_cols, figsize=[20*num_rows, 5*num_cols])

    for ax_number, (clf_name, cm) in enumerate(confusion_matrices.items()):
        # Turn off initial axis
        ax[ax_number].axis('off')
        # Generate CM title
        title = f'Confusion Matrix \n {clf_name}'
        # Get subplot index and plot CM for given index
        index = int(str(num_rows) + str(num_cols) + str(ax_number + 1))
        plot_confusion_matrix(cm, classes=y_preds[clf_name].unique(), fig=fig, index=index, title=title)
        
        
def plot_binary_clf_histogram(y_test, y_pred, bins=50, normalize=True, fig=None, index=111, title='Model predictions (histograms of positive and negative classes)', categorical_palette=sns.color_palette()):
    '''
    Plot histograms of model-predicted probability counts for both classes
    
    Parameters
    ----------
    y_test : pandas.Series
        Test-set labels
    y_pred : pandas.Series
        Test-set model predictions
    bins : int
        Number of bins in each histogram
    normalize : bool
        Whether to display absolute or relative counts (useful for visualising when big 0/1 class imbalance)
    '''
    
    if fig is None:
        fig = plt.figure(figsize=[20, 10])
    ax = fig.add_subplot(index)
    
    negatives = y_pred[y_test == False]
    positives = y_pred[y_test == True]
    
#     plt.figure(figsize=[20, 10])
    sns.set_palette(categorical_palette)
    sns.distplot(negatives, hist=True, kde=False, norm_hist=normalize, bins=bins, ax=ax)
    sns.distplot(positives, hist=True, kde=False, norm_hist=normalize, bins=bins, ax=ax)
    ax.set_xlabel('Model predicted probability of positive class', fontsize='x-large'), 
    ax.set_ylabel('Counts (of binned probability)', fontsize='x-large')
    ax.set_title(title, fontsize='xx-large')

    
def plot_binary_clf_histograms(y_test, y_scores):
    '''
    Plots a series of histograms from a set of y predictions
    
    Parameters
    ----------
    confusion_matrices : dict
        Dict of confusion matrix arrays
    '''
    
    y_scores_models = y_scores.drop('LABELS', axis=1)
    
    num_rows, num_cols = get_subplot_dims(y_scores_models.shape[1], max_cols=1)

    fig, ax = plt.subplots(num_rows, num_cols, figsize=[20, 10*num_rows])

    for ax_number, (clf_name, y_score) in enumerate(y_scores_models.iteritems()):
        # Turn off initial axis
        ax[ax_number].axis('off')
        # Generate CM title
        title = f'Model Predictions \n {clf_name}'
        # Get subplot index and plot CM for given index
        index = int(str(num_rows) + str(num_cols) + str(ax_number + 1))
        plot_binary_clf_histogram(y_test, y_score, title=title, fig=fig, index=index)