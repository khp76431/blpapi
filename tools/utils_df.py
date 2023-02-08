import numpy as np
import pandas as pd
import datetime
from tools import utils


def add_new_columns(df, columns, data):
    # Add empty columns.
    df = df.reindex(columns=df.columns.tolist() + columns)
    # Assign to empty columns.
    df[columns] = data
    # Return the DataFrame.
    return df


def keep_first_or_last_day(df, keep_first=False, method='ffill', h_map=None):
    """

    :param df:              A DataFrame with daily data.
    :param keep_first:      A flag deciding if the first or the last day of a period is to be used.
    :param method:          Method to use with fillna.
    :param h_map:           Map the index to a period. Default is to map the index to 'YYYY-MM'.
    :return:                DataFrame.
    """
    # Temporary dumm column name.
    TEMP = 'TEMP'

    # Assert that df is a DataFrame.
    assert isinstance(df, pd.DataFrame)

    # Assert that method is valid.
    assert method in {'backfill', 'bfill', 'pad', 'ffill', None}

    # Get the map-function.
    h_map = _get_map_function(df.index[0], h_map)

    # Fill nans.
    if method is not None:
        df.fillna(method=method, inplace=True)

    # Create a temporary column to be used by the drop_duplicate function below.
    df[TEMP] = df.index.map(h_map)

    # Construct the keep argument. Note that if keep = False then ALL duplicates are removed (which is not really the
    # purpose of this function.
    keep = False if keep_first is None else ('first' if keep_first else 'last')

    # Drop duplicates. Special care is needed if the columns of the DataFrame is a multi-index.
    if isinstance(df.columns, pd.core.indexes.multi.MultiIndex):
        # This is a hack to be able to drop duplicates from a DataFrame with multi-index columns. But it works...
        # Record old column names.
        columns = df.columns
        # Temporarily change column names.
        num_cols = df.shape[1]
        df.columns = range(num_cols)
        # Drop duplicates from the last column.
        df.drop_duplicates(subset=[num_cols - 1], keep=keep, inplace=True)
        # Restore previous column names.
        df.columns = columns
    else:
        df.drop_duplicates(subset=[TEMP], keep=keep, inplace=True)

    # Delete the TEMP column.
    del df[TEMP]

    # Return the DataFrame.
    return df


def _get_map_function(test_value, h_map):
    if h_map is None:
        # Initiate the error message.
        msg = None
        # Define a default map function.
        if isinstance(test_value, (datetime.datetime, datetime.date, pd.Timestamp)):
            h_map = lambda x: x.strftime('%Y-%m')
        elif isinstance(test_value, str):
            if len(test_value) == 10 and test_value[4] == '-' and test_value[7] == '-':
                h_map = lambda x: x[:7]
            else:
                msg = 'Unable to suggest default map function for index elements of str type which is not of the form ' \
                      'YYYY-MM-DD'
        else:
            msg = 'Unable to suggest default map function for index elements of this type: ' + type(test_value).__name__
        if msg is not None:
            raise Exception(msg)
        #
        is_default = True
    else:
        is_default = False

    try:
        test_result = h_map(test_value)
        # Well done the test was successful.
    except Exception as err:
        if is_default:
            msg = '[ERROR] The default map function was not able to operate on the index.'
        else:
            msg = '[ERROR] The supplied map function was not able to operate on the index.'
        print(msg)
        raise Exception(err)

    # Return the map function.
    return h_map


def concat_dfs_from_list(df_list, keys=None,  names=None, axis=1):
    # Assert that df_list is indeed a list.
    assert isinstance(df_list, list)
    # Assert that all elements in df_list are DataFrames.
    assert all([isinstance(df_list[i], pd.DataFrame) for i in range(len(df_list))])

    # Concatenate the DataFrames in df_list.
    df_cnt = pd.concat(df_list, axis=axis)

    if keys is not None:

        # Number of columns for the elements in df_list.
        _num_cols = df_list[0].shape[1]

        # Assert that keys are appropriate...
        assert (isinstance(keys, (list, tuple)) and len(keys) == len(df_list))
        # Assert that all elements in df_list has the same number of columns.
        assert all([df_list[i].shape[1] == _num_cols for i in range(1, len(df_list))])
        # Assert that the column names are the same across all elements in df_list.
        assert all([df_list[i].columns.names == df_list[0].columns.names for i in range(1, len(df_list))])
        # Assert that the columns are the same across all elements in df_list.
        assert all([all(df_list[i].columns == df_list[0].columns) for i in range(1, len(df_list))])
        # Get the column names.
        names = df_list[0].columns.names if names is None else names

        # Build new tuple for the multi-index column.
        # Clean the keys.
        keys = [utils.get_clean_column_name(k) for k in keys]
        cols_0 = utils.get_flatten_list([[k]*_num_cols for k in keys])
        cols_1 = df_list[0].columns.tolist()*len(keys)
        tuples = list(zip(cols_0, cols_1))
        # Change the column names.
        df_cnt.columns = pd.MultiIndex.from_tuples(tuples, names=names)

    # Return the concatenated DataFrame.
    return df_cnt


def drop_initial_nans(df):
    #
    # This function drops initial rows where ALL data points are nans.
    #
    # Get the first index where all values in a row are valid data points.
    if isinstance(df.columns, pd.core.indexes.multi.MultiIndex):
        df = _drop_initial_nans_multi(df)
        return df

    try:
        # idx = [all(x) for x in ~np.isnan(df).values].index(True)
        idx = [all(x) for x in np.isnan(df).values].index(False)
    except:
        # If an exception has been raised, there are one or more (possibly all) series in df which consists of
        # only nans. We must therefore ignore these series.
        try:
            idx = [all(x) for x in ~np.isnan(df.dropna(axis=1, how='all')).values].index(True)
        except:
            # df contains only nans?
            return df

    # Returned a sliced DataFrame.
    return df.iloc[idx:, :]


def _drop_initial_nans_multi(df):

    idx = 0
    tickers = df.columns.get_level_values(0).tolist()
    for ticker in tickers:
        idx = max(idx, [all(x) for x in ~np.isnan(df[ticker].dropna(axis=1, how='all')).values].index(True))

    return df.iloc[idx:, :]


def drop_nans(*dfs, fill_method='bfill', how='any'):
    # Assert that all dfs have columns of type pd.core.indexes.base.Index.
    assert all([isinstance(df.columns, pd.core.indexes.base.Index) for df in dfs])
    # Concatenate dfs.
    df = pd.concat(dfs, axis=1)
    # Change columns names.
    _row_0 = utils.get_flatten_list([[i]*dfs[i].shape[1] for i in range(len(dfs))])
    _row_1 = utils.get_flatten_list([dfs[i].columns.tolist() for i in range(len(dfs))])
    df.columns = pd.MultiIndex.from_tuples(zip(_row_0, _row_1))
    # Fill nans.
    if fill_method is not None:
        df.fillna(method=fill_method, inplace=True)
    # Drop initial nans.
    df.dropna(how=how, inplace=True)
    # Return the individual dfs.
    return [df[i] for i in range(len(dfs))]


def fun_on_df(df, h_funs=None):

    assert h_funs is None or type(h_funs).__name__=='function' or isinstance(h_funs, tuple)

    if h_funs is None:
        return df

    if isinstance(h_funs, tuple):
        #
        # Iterate over the elements in h_funs, which must be a tuple (of tuples) with the following structure:
        #
        #       h_funs = ((h_f_1, args_1), (h_f_2, args_2), ... (h_f_n, args_n)), i.e.
        #
        # i.e. each sub-tuple consists of a function handle and a dict of arguments to be used as
        #
        #       df = h_f_i(df, args_i).
        #
        # It is therefore important that any inplace arguments is set to False, since by definition in this structure,
        # the DataFrame IS returned.
        #
        for h_fun, args in h_funs:
            # If inplace is an argument, then set it to False (since the DataFrame is returned).
            if 'inplace' in args:
                args['inplace'] = False
            # If h_fun is a string, then it is assumed to be a method of the pd.DataFrame class. This is tested first,
            # and if the method exists h_fun is then set to be that method, rather than its name.
            if isinstance(h_fun, str):
                if hasattr(pd.DataFrame, h_fun):
                    h_fun = getattr(pd.DataFrame, h_fun)
                else:
                    raise Exception('Unable to interpret this str as a function (it is not a method of a DataFrame): ' + h_fun)
            try:
                # NOTE: The DataFrame IS returned here.
                df = h_fun(df, **args)
            except Exception as err:
                print('Unable to run method of DataFrame')
                print(str(err))
    else:
        # In this case h_funs is a function, which is then executed here.
        df = h_funs(df)

    # Return the DataFrame.
    return df


def clean_columns(df, remove_ticker_type=True, remove_chars=True):
    # Clean the tickers in the columns.
    tickers = [utils.get_clean_ticker(x, remove_ticker_type=remove_ticker_type, remove_chars=remove_chars) for x in
               df.columns.get_level_values(0).values.tolist()]
    # Build new tuple for the multi-index column.
    tuples = list(zip(tickers, df.columns.get_level_values(1)))
    # Change the column names.
    df.columns = pd.MultiIndex.from_tuples(tuples, names=df.columns.names)
    # Return the DataFrame.
    return df


"""
def read_dataframe_from_csv(filepath, sep=', ', delimiter=None, header=0, names=None, index_col=None,
                            usecols=None, squeeze=False, prefix=None, mangle_dupe_cols=True, dtype=None, engine=None,
                            converters=None, true_values=None, false_values=None, skipinitialspace=False, skiprows=None,
                            nrows=None, na_values=None, keep_default_na=True, na_filter=True, verbose=False,
                            skip_blank_lines=True, parse_dates=False, infer_datetime_format=False, keep_date_col=False,
                            date_parser=None, dayfirst=False, iterator=False, chunksize=None, compression='infer',
                            thousands=None, decimal=b'.', lineterminator=None, quotechar='"', quoting=0,
                            escapechar=None,
                            comment=None, encoding=None, dialect=None, tupleize_cols=None, error_bad_lines=True,
                            warn_bad_lines=True, skipfooter=0, doublequote=True, delim_whitespace=False,
                            low_memory=True,
                            memory_map=False, float_precision=None,
                            start_date=None, end_date=None, index_type=None, add_mid=False, clean_column_names=False):
     
    # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html#pandas.read_csv

     

     
    # df = pd.read_csv(filepath, sep=sep, delimiter=delimiter, header=header, names=names, index_col=index_col,
    #                 usecols=usecols, squeeze=squeeze, prefix=prefix, mangle_dupe_cols=mangle_dupe_cols, dtype=dtype,
    #                 engine=engine, converters=converters, true_values=true_values, false_values=false_values,
    #                 skipinitialspace=skipinitialspace, skiprows=skiprows, nrows=nrows, na_values=na_values,
    #                 keep_default_na=keep_default_na, na_filter=na_filter, verbose=verbose,
    #                 skip_blank_lines=skip_blank_lines, parse_dates=parse_dates,
    #                 infer_datetime_format=infer_datetime_format, keep_date_col=keep_date_col, date_parser=date_parser,
    #                 dayfirst=dayfirst, iterator=iterator, chunksize=chunksize, compression=compression,
    #                 thousands=thousands, decimal=decimal, lineterminator=lineterminator, quotechar=quotechar,
    #                 quoting=quoting, escapechar=escapechar, comment=comment, encoding=encoding, dialect=dialect,
    #                 tupleize_cols=tupleize_cols, error_bad_lines=error_bad_lines, warn_bad_lines=warn_bad_lines,
    #                 skipfooter=skipfooter, doublequote=doublequote, delim_whitespace=delim_whitespace,
    #                 low_memory=low_memory, memory_map=memory_map, float_precision=float_precision)
     
    df = pd.read_csv(filepath, index_col=index_col, parse_dates=parse_dates,
                     infer_datetime_format=infer_datetime_format)

    if isinstance(df.index.values[0], np.datetime64):
        df = df[[not np.isnat(x) for x in df.index.values]]

    if start_date:
        if not isinstance(start_date, type(df.index[0])):
            start_date = datetime_utils.get_datetime(start_date, output_type=type(df.index[0]))
        df = df[df.index >= start_date]

    if end_date:
        if not isinstance(end_date, type(df.index[0])):
            end_date = datetime_utils.get_datetime(end_date, output_type=type(df.index[0]))
        df = df[df.index <= end_date]

    if add_mid:
        raise Exception('Cannot add Mid series, this is not yet implemented.')
        # df = add_mid_series(df)

    if index_type:
        if not type(df.index[0]) == index_type:
            df.index = [*map(lambda x: datetime_utils.get_datetime(x, output_type=index_type), df.index)]

    if clean_column_names:
        df.rename(columns=dict(zip(df.columns, [*map(str.strip, df.columns)])), inplace=True)

    if usecols:
        df = df[usecols]

    return df
"""