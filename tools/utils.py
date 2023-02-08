import numpy as np
import pandas as pd
from tools import utils_help as uh
import datetime

BAD_CHARS = ':%?'
POST_FIX = ('CURNCY', 'COMDTY', 'INDEX')

def get_flatten_list(list_of_list, output_type=list):
    """
    This function flattens a list of lists into a one-dimensional list, tuple or np.ndarray.
    :param      list_of_list:
    :param      output_type: The type of the returned flattened 'list'.
    :return:    a flattened list, tuple or np.ndarray.

                    Simple example.
    Example 1.      flatten_list([[1, 2], [3, 4, 5]])
                    returns [1, 2, 3, 4]

                    Specify output_type.
    Example 2a.     flatten_list([[1, 2], [3, 4]], output_type=tuple)
                    flatten_list([[1, 2], [3, 4]], output_type=np.array)
                    flatten_list([[1, 2], [3, 4]], output_type=np.ndarray)
    """
    # Input must be a list of list (for now).
    assert isinstance(list_of_list, list)

    # Get the output type. This is either np.ndarray, list or tuple.
    output_type = uh.get_output_type(output_type)

    output = [item for sublist in list_of_list for item in sublist]

    if output_type == list:
        return output
    elif output_type == tuple:
        return tuple(output)
    elif output_type == np.ndarray:
        return np.array(output)
    else:
        raise TypeError('output_type must be list, tuple or an np.ndarray.')


def get_array(value, output_type=np.ndarray, allowed_types=None, allow_none=False, dtype=None, h_fun=None):
    """
    This function returns either a np.bdarray, list or a tuple of the passed value. It can thus be used to transform
    between these types, and to 'array-ify' scalar values.
    Operations on the elements in value can be carried out by invoking the h_fun argument (see Examples 4b and 5 below).
    :param value: to be converted into a np.ndarray, list or a tuple.
    :param output_type: np.ndarray (or np.array), list or a tuple.
    :param allowed_types: type (or tuple of types) of value or elements of value. Examples: int, float etc.
    :param allow_none: Flag to decide whether or not a value of None or nan is accepted.
    :param dtype: If dtype is passed, then the output it created using np.array(x, dtype=dtype) before possibly converting to list or tuple.
    :param h_fun: handle to a function which will operate on all elements in value.
    :return: an np.ndarray, list or a tuple of value.

    Example 1:      Create a np.ndarray from a tuple.
                    get_array(5, output_type=np.ndarray)
                    returns: np.array([5])

    Example 2       Create a list from a np.ndarray.
                    get_array(np.array([1, 2, None, 3]), allow_none=True, output_type=tuple)
                    returns: (1, 2, None, 3)

    Example 3a:     Create a tuple from a test date.
                    get_array(datetime.date(2018, 9, 12), output_type=tuple)
                    returns (datetime.date(2018, 9, 12),)

    Example 3b:     Create a tuple from a test date. This will produce an error.
                    get_array(datetime.date(2018, 9, 12), output_type=tuple, allowed_types=datetime.datetime),
                    whereas
                    get_array(datetime.date(2018, 9, 12), output_type=tuple, allowed_types=datetime.date)
                    will work.

    Example 4a:     Create a np.ndarray of mixed types.
                    get_array([3.0, 5, 'a'], dtype=object)
                    returns array([3.0, 5, 'a'], dtype=object)

    Example 4b:     Create a np.ndarray of mixed types where all strings are replaced with nan.
                    get_array([3.0, 5, 'a'], allow_none=True, h_fun=lambda x: np.nan if isinstance(x, str) else x)
                    returns array([ 3.,  5., nan])

    Example 5:      To create a list where all values have been multiplies by 2, and ignoring all None.
                    get_array([2, 3, 4, None], output_type=list, allow_none=True, h_fun=lambda x: 2 * x)
                    returns [4, 6, 8, None]
    """

    # Get the output type. This is either np.ndarray, list or tuple.
    output_type = uh.get_output_type(output_type)

    # If value is None or nan, then do this special case first.
    if uh.is_none_or_nan(value):
        if allow_none:
            value = uh.convert_value([value], output_type)
            return value
        else:
            raise TypeError('value is None (or nan) but allow_none is False.')

    # Ensure that allowed_types is a tuple and that its elements are allowed.
    if allowed_types is not None:
        # If allowed_types is not None, ensure it is a tuple.
        allowed_types = (allowed_types,) if not isinstance(allowed_types, tuple) else allowed_types
        # Assert that allowed_types is a typle.
        assert isinstance(allowed_types, tuple)
        # Assert that all elements in allowed_types are of type type.
        assert all([type(x) == type for x in allowed_types])

    # For np.ndarray do this.
    if type(value) == np.ndarray:
        # If value is of type ndarray, then we must include the numpy types for int and float in allowed_types.
        if allowed_types is not None:
            allowed_types = uh.add_numpy_types(allowed_types)
        # Ensure value is a 'sized object', e.g. array(5) -> array([5]).
        # NOTE: len(array(5)) fails, whereas len(array([5])) = 1.
        if len(value.shape) == 0:
            value = np.array([value])

    if dtype is not None:
        # If dtype has been passed, then use the function array(x,dtype=dtype) first to convert all elements in the
        # array to the type dtype. Raise a TypeError if this fails.
        try:
            value = np.array(value if uh.is_array_class(value) else [value], dtype=dtype)
        except TypeError:
            raise TypeError('Could not convert the value to an array with dtype ' + dtype.__name__ + '.')

    # Redefine h_fun if allow_none is True.
    if h_fun is not None and allow_none:
        _temp = h_fun
        h_fun = lambda x: x if uh.is_none_or_nan(x) else _temp(x)

    # In this case dtype is None then first convert value to output_type.
    value = uh.convert_value(value if uh.is_array_class(value) else [value], output_type, h_fun=h_fun)

    # Ensure that all elements are of allowed types, and whether there are any Nones or nans.
    uh.test_nones_and_types(value, allow_none, allowed_types)

    # Return the value.
    return value


def get_col_names(col_names, ref_columns, output_type=None, level=None):
    """
    This function identifies the entries in col_names in ref_columns, and returns the corresponding identified names.
    It is useful for finding column names in DataFrames. E.G. If there is a column in a DataFrame with the name 'Test',
    but it might be called 'TEST', 'Test' or 'test', then the proper (i.e. case sensitive) name can be found by

            prop_name = get_col_names('test', df)

    which is identical to

            prop_name = get_col_names('TEST', df.columns).

    :param col_names: a str or sequence of strings of column names to be identified in ref_columns.
    :param ref_columns: a sequence of reference column names, or a DataFrame.
    :param output_type
    :return: a sequence of column names identified in ref_columns.


                    Simple lookup. col_names is either a str, tuple or a list, and ref_columns is either a tuple or a list.
                    Note that the output inherits the type of the input.
    Example 1a:     get_col_names('aa', ('Aa', 'Bb', 'Cc'))
                    returns 'Aa'

    Example 1b:     get_col_names(('aa', 'bb'), ('Aa', 'Bb', 'Cc'))
                    returns ('Aa', 'Bb')

    Example 1c:     get_col_names(['aa', 'bb'], ('Aa', 'Bb', 'Cc'))
                    returns ['Aa', 'Bb']


                    Specify the type of the output.
    Exemple 2a      get_col_names('aa', ('Aa', 'Bb', 'Cc'), output_type=list)
                    returns ['Aa']

    Example 2b:     get_col_names(['aa', 'bb'], ['Aa', 'Bb', 'Cc'], output_type=tuple)
                    returns ('Aa', 'Bb')


                    ref_columns from DataFrame, or DataFrame.columns
    Example 3:      df = pd.DataFrame(data=[[1, 2, 3]], ref_columns=['ColA', 'ColB', 'colC'])
                    get_col_names('cola', df)
                    return 'ColA' (a string)

    Example 4:      df = pd.DataFrame(data=[[1, 2, 3]], ref_columns=['ColA', 'ColB', 'colC'])
                    get_col_names(('cola',), df.columns)
                    return ('ColA',)

    Example 5:      df = pd.DataFrame(data=[[1, 2, 3]], ref_columns=['ColA', 'ColB', 'colC'])
                    get_col_names(('cola', 'colb'), df, output_type=list)
                    ['ColA', 'ColB']
    """

    # Set a flag to determine if a str has been passed.
    input_type = type(col_names)

    # Get an array of col_names which is upper-case.
    col_names = get_array(col_names, output_type=list, h_fun=str.upper)

    # Check if ref_columns is a DataFrame or the index of a DataFrame. In these cases, transform ref_columns to a list.
    if isinstance(ref_columns, pd.DataFrame):
        if isinstance(ref_columns.columns, pd.core.indexes.multi.MultiIndex):
            if level is not None:
                ref_columns = ref_columns.columns.get_level_values(level).tolist()
            else:
                raise Exception('The columns of the DataFrame is of type MultiIndex. A level must be supplied to find reference columns.')
        else:
            ref_columns = ref_columns.columns.tolist()
    elif isinstance(ref_columns, pd.core.indexes.base.Index):
        ref_columns = ref_columns.tolist()

    # Assert that ref_columns is either a np.ndarray, list or tuple.
    assert uh.is_array_class(ref_columns)

    # Transform ref_columns to a list.
    if not isinstance(ref_columns, list):
        ref_columns = list(ref_columns)

    # Create a upper-case version of ref_columns for lookup purposes (using the index method).
    columns_up = [cn.upper() for cn in ref_columns]

    # Assert that the upper-case version of ref_columns has unique entries.
    assert len(set(columns_up)) == len(columns_up)

    # Assert that all col_names are in columns_up.
    assert all([*map(lambda cn: cn in columns_up, col_names)])

    # Find the corresponding matched column name in ref_columns.
    output = [*map(lambda cn: ref_columns[columns_up.index(cn)], col_names)]

    # Return the column name.
    if output_type is None:
        if input_type == str:
            return output[0]
        else:
            return input_type(output)
    else:
        return output_type(output)


def get_unique_list(_l):
    #    
    _ul = list()
    # Iterate over the elements in _l and add them to _ul if they are not already in _ul
    for s in _l:
        if s not in _ul:
            _ul.append(s)
    # Return the list of unique elements
    return _ul


def get_yest_cob():
    _now = datetime.datetime.now().date()
    _wkday = _now.weekday()
    _now -= datetime.timedelta(days=3 if _wkday==0 else 2 if _wkday==6 else 1)
    #
    return _now


def get_clean_ticker(ticker, remove_ticker_type=True, remove_chars=True):
    # Remove any Index, Curncy, Comdty...
    if remove_ticker_type:
        for pf in POST_FIX:
            ticker = ticker.upper().replace(pf.upper(), '').strip()
    # Remove any funny character ($?%...)
    if remove_chars:
        ticker = get_clean_column_name(ticker)
        # Return the cleaned ticker.
    return ticker


def get_clean_column_name(col_name):
    #
    for char in BAD_CHARS:
        col_name = col_name.replace(char, '')
    return col_name


def main():
    # import doctest
    # doctest.testmod()
    print(get_array.__doc__)


if __name__ == '__main__':
    main()