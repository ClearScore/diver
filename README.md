# DIVER

`Diver` is the `D`ataset `I`nspector, `V`isualiser and `E`ncode`r` library, automating and codifying common data science project steps as standardised and reusable methods.

<p align="center">
  <img width="460" height="300" src="pictures/stingray.jpg">
</p>

# dataset-conditioner

![`fit_transform` flow](pictures/readme_flow.png)

## Procedure

See `example-notebooks/house-price-demo.ipynb` for a full walkthrough

## Future Work

### `categorical_excess_cardinality_flagger_and_reducer`
- Option for instances where there are no categorical features

### `missing_value_conditioner`
- Choose between either {use means from train set (default), calculate means for test set}
- Missing values for categorical features

- Implement this: https://measuringu.com/handle-missing-data/

- GOOD READING: https://towardsdatascience.com/6-different-ways-to-compensate-for-missing-values-data-imputation-with-examples-6022d9ca0779

### `ordinal_encoder`
- Create a function to do this

### `timestamp_encoder`
- is_public_holiday : bool
- Update above diagram

### Remove warnings

### Make robust to non-consecutive indices in input df

### Unit test all functions

### PCA option?
- https://medium.com/apprentice-journal/pca-application-in-machine-learning-4827c07a61db

## Useful info

# dataset_inspector
- https://towardsdatascience.com/smarter-ways-to-encode-categorical-data-for-machine-learning-part-1-of-3-6dca2f71b159
### Label balanced class checker (for classification problems)
- https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/

### Extreme values

# dataset_visualiser

### Distribution and correlation analysis
- Display correlation matrix for top `n` correlates alongside target at the bottom
- Display pairplot for top `n` correlates alongside target at the bottom
- Or instead of `top n` correlates, instead threshold of `cumulative variance`
- Option to DROOP lower correlates (lower than threshold) if desired

# Useful reading
https://machinelearningmastery.com/process-for-working-through-machine-learning-problems/


- Option for PCA at this stage?


