def _get_boolean_elements():
    '''
    Returns sets of elements recognisable as booleans
    '''
    
    # Inputs to be recognised as booleans
    true_set = {'True','true','TRUE','tru',1,'t','T','1', float(1), True, 'yes', 'Yes', 'YES', 'Y', 'y'}
    false_set = {'False','false','FALSE','fals',0,'f','F','0', float(0), False, 'no', 'No', 'NO', 'N', 'n'}
    
    return true_set, false_set