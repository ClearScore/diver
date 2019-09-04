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
- A function which does a value count and then keeps only the top n categories and assigns all else to an `other` category

### `timestamp_encoder`
- is_public_holiday : bool
- Update above diagram

### `boolean_encoder`
- Recognise `int` as `bool`

### Remove warnings

### Make robust to non-consecutive indices in input df

### Unit test all functions

## Useful info
