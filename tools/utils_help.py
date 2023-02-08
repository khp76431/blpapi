import numpy as np


def get_output_type(output_type):

    # Asset that output_type is wither None or a type.
    assert output_type is None or isinstance(output_type, type)

    # Ensure that the output_type defaults to np.ndarray.
    if output_type is None:
        output_type = np.array
    # Do nothing if output_type is an np.ndarray, list or tuple.
    elif output_type in set([np.ndarray, list, tuple]):
        # Do nothing here.
        pass
    else:
        # For any other output_type, raise and error.
        raise ValueError(
            'output_type must be either an np.ndarray, list or a tuple. This output_type is not allowed: ' +
            output_type.__name__ + '.')

    # Return the output_type.
    return output_type


def is_none_or_nan(x):
    return x is None or (isinstance(x, (float, np.floating)) and np.isnan(x))


def is_array_class(value):
    return isinstance(value, (np.ndarray, list, tuple))


def is_allowed_type(x, allowed_types):
    return True if allowed_types is None else isinstance(x, allowed_types)


def add_numpy_types(allowed_types):
    # Possibly add np.integer to allowed_types.
    if int in allowed_types:
        allowed_types = (*allowed_types, np.integer)
    # Possibly add np.floating to allowed_types.
    if float in allowed_types:
        allowed_types = (*allowed_types, np.floating)
    # Return adjusted allowed_types.
    return allowed_types


def convert_value(value, output_type, h_fun=None):
    # We know that value is either an ndarray, list or a tuple at this point.

    #
    if h_fun is not None:
        value = [h_fun(x) for x in value]

    if output_type == np.ndarray:
        # Convert to array.
        if not isinstance(value, np.ndarray):
            value = np.array(value)

    elif output_type == list:
        # Convert to list.
        if not isinstance(value, list):
            value = list(value)

    elif output_type == tuple:
        # Convert to tuple.
        if not isinstance(value, tuple):
            value = tuple(value)

    # Return the converted value.
    return value


def test_nones_and_types(value, allow_none, allowed_types):
    # Ensure that all elements are of allowed types, and whether there are any Nones or nans.
    if not all([allow_none if is_none_or_nan(x) else is_allowed_type(x, allowed_types) for x in value]):
        _cts_dis_alw_none = not allow_none and any([is_none_or_nan(x) for x in value])
        _cts_dis_alw_types = any([not is_allowed_type(x, allowed_types) for x in value])
        msg = None
        if _cts_dis_alw_none:
            msg = 'value contains Nones or nans, which is not allowed.'
        if _cts_dis_alw_types:
            _msg = 'value contains elements of types which are not allowed.'
            msg = msg[:-1] + ', and ' + _msg if _cts_dis_alw_none else _msg
        raise Exception(msg)






"""
def contains_digits_only(chrs):
    return all([*map(lambda ch: ch in set(const.DIGITS), chrs)])

def is_month_name(name):
    return len(name) == 3 and name.upper() in set(const.MONTH_NAMES)
    
def get_delimeter(str_date):
    _delims = list(set([ch for ch in str_date if ch in set(const.DATE_DELIMETERS)]))
    if len(_delims) == 0:
        delim = ''
    elif len(_delims) == 1:
        delim = _delims[0]
    else:
        raise Exception(
            'There are different types of delimeters in str_date. This str_date is not allowed: ' + str_date + '.')
    return delim
 

def check_for_nones_or_nans(value):
    if any(map(is_none_or_nan, value)):
        raise TypeError('There are one or more Nones or nans in the array, but allow_none is False.')
 
def raise_allowed_type_error(idx, allowed_types):
    if isinstance(allowed_types[idx.index(False)], str):
        msg = 'allowed_types must contain class names (types) such as int, float. This value is not allowed: ' \
          + "'" + allowed_types[idx.index(False)] + "'" + '.'
    else:
        msg = 'allowed_types must contain class names (types) such as int, float...'
    raise TypeError(msg)
    
def _get_class_name(output_type):
    return 'ndarray' if output_type == np.array else ('list' if output_type == list else 'tuple')


    
def _get_error_string_check_types(output_type, disallowed_type_names, allowed_type_names):
    msg = 'Unable to create ' + _get_class_name(
        output_type) + '. The elements in value has the following disallowed type' + (
              's: ' if len(disallowed_type_names) > 1 else ': ') + ', '.join(
        disallowed_type_names) + '. Allowed type' + ('s are: ' if len(allowed_type_names) > 1 else ' is: ') + ', '.join(
        allowed_type_names) + '.'
    return msg
    
def check_types(value, output_type, allowed_types):
    allowed_type_names = [*map(lambda tpe: tpe.__name__, allowed_types)] if isinstance(allowed_types,
                                                                                       tuple) else allowed_types.__name__
    disallowed_type_names = [type_name for type_name in [x for x in set(map(lambda y: type(y).__name__, value))] if
                             type_name not in allowed_type_names]
    raise TypeError(_get_error_string_check_types(output_type, disallowed_type_names, allowed_type_names))
 

def get_date_format_raise_error(date_string):

    num_chars = len(date_string)

    if num_chars == 6:
        msg = 'If date_string has six characters but not all characters are numbers.'
    elif num_chars == 7:
        msg = 'If date_string has seven characters then the two first and two last must be digits, and the middle three ' \
              'must be a valid month name.'
    elif num_chars == 8:
        msg = None
    elif num_chars == 9:
        msg = None

    msg = str(num_chars)+ ' A date of this format is not allowed: ' + date_string + '.'

    raise Exception(msg)
    
"""