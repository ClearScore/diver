# dataset-conditioner

![`fit_transform` flow](pictures/readme_flow.png)

## Procedure

See `demo.ipynb` for a full walkthrough

## Future Work

### `missing_value_conditioner`
- Choose between either {use means from train set (default), calculate means for test set}
- Missing values for categorical features
- Check % of missing values in each column and drop feature (and store) if above threshold

- Implement this: https://measuringu.com/handle-missing-data/

### `ordinal_encoder`
- Create a function to do this

### `nominal_encoder`
- A function which does a value count and then keeps only the top n categories and assigns all else to an `other` category - see also `dataset_inspector` below

### `timestamp_encoder`
- is_public_holiday : bool
- Update above diagram

### `boolean_encoder`
- Recognise `int` as `bool`
- Fix error where it crashes if all either `True` or `False`

### Remove warnings

### Make robust to non-consecutive indices in input df

### Unit test all functions

### PCA option?
- https://medium.com/apprentice-journal/pca-application-in-machine-learning-4827c07a61db

## Useful info

# dataset_inspector

### Categorical cardinality inspector
- Past a threshold, either DROP column or assign smaller classes to `other`
- https://towardsdatascience.com/smarter-ways-to-encode-categorical-data-for-machine-learning-part-1-of-3-6dca2f71b159

### Distribution and correlation analysis
- Display correlation matrix for top `n` correlates alongside target at the bottom
- Display pairplot for top `n` correlates alongside target at the bottom
- Or instead of `top n` correlates, instead threshold of `cumulative variance`
- Option to DROOP lower correlates (lower than threshold) if desired
- Option for PCA at this stage?

### Missing values inspector
- Past a threshold, either DROP column or FLAG features with an unacceptable % of missing values

### Label balanced class checker (for classification problems)

### Extreme values
