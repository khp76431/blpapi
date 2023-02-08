from tools import utils_dt
from tools import utils_df
from tools import utils
import datetime
import pandas as pd


class DataProviderBase(object):

    def __init__(self, name):

        # Set the name of the DataProvider.
        self._name = name

        # Keyword map, which is updated in each sub-class.
        self._kwrd_map = {'ENDDATE': 'end_date', 'END_DATE': 'end_date', 'END': 'end_date',
                          'STARTDATE': 'start_date', 'START_DATE': 'start_date', 'START': 'start_date'}

        self._def_start_date = datetime.date(1990, 1, 1)

        # See below for a description of self._h_fun.
        self._h_fun = None

    def _do_pre(self, kwargs):

        #
        kwargs = self._conform_keywords(kwargs)

        # Set the self._h_fun property.
        self._h_fun = kwargs.pop('h_fun', None)

        # Return the kwargs.
        return kwargs

    def _do_post(self, df, index_type, harmonize=False, delete_weekends=True):

        # Delete weekends.
        if delete_weekends:
            df = self._delete_weekends(df)

        # Change the type of the index.
        if index_type is not None:
            df.index = [utils_dt.get_datetime(x, output_type=index_type) for x in df.index.values]

        # Kill the name of the index.
        df.index.name = None

        # Drop initial nans.
        if harmonize:
            df = utils_df.drop_initial_nans(df)

        # Iterate over the elements in self._h_fun, which must be a tuple (of tuples).
        df = utils_df.fun_on_df(df, h_funs=self._h_fun)

        # Return the DataFrame.
        return df

    def _conform_keywords(self, kwargs):

        keywords = list()
        values = list()

        # Record the keywords and their associated values in kwargs IF they are in the keyword map.
        for k, v in kwargs.items():
            if k.upper() in self._kwrd_map:
                keywords.append(k)
                values.append(v)


        for k, v in zip(keywords, values):
            del kwargs[k]
            kwrd = self._kwrd_map.get(k.upper())
            kwargs[kwrd] = self._kval_map.get(kwrd).get(v, v) if kwrd in self._kval_map else v

        # Return the dictionary.
        return kwargs

    def _get_data(self, h_get_data, securities, batch_size, info_on_screen=False, screen_msg=None):
        
        # Ensure that the securities are unique.
        _num_els = len(securities)
        securities = utils.get_unique_list(securities)
        if _num_els != len(securities):
            print('[WARNING] Duplicate securities have been removed.')
          
        # batches. Split the list of securities according to the batch_size.
        stt_idx, end_idx = self._get_batch_indices(securities, batch_size)

        # Initiate a list to contain DataFrames for the different batches.
        _num_batches = len(stt_idx)
        df_temp = [None] * _num_batches

        # Iterate over all batches.
        for i, (si, ei) in enumerate(zip(stt_idx, end_idx)):
            if info_on_screen:
                _info = '' if screen_msg is None else ' [' + screen_msg + ']'
                print('Now doing ' + self._name + ' batch ' + str(i + 1) + ' of ' + str(_num_batches) + _info + '...',
                      end="", flush=True)
            try:
                df_temp[i] = h_get_data(securities[si: ei])
            except Exception as err:
                print('Error getting data. Error message: ' + str(err))
                raise err
            if info_on_screen:
                print(' Done.')

        # Concatenate the DataFrames from the different batches.
        df = pd.concat(df_temp, axis=1)
         
        # Return the DataFrame.
        return df

    @staticmethod
    def _get_batch_indices(securities, batch_size):
        if batch_size is None:
            stt_idx = [0]
            end_idx = [len(securities)]
        else:
            _num = int((len(securities)-1)/batch_size)
            stt_idx = [batch_size * r for r in range(_num+1)]
            end_idx = [x+batch_size for x in stt_idx[:-1]] + [len(securities)]
        #
        return stt_idx, end_idx

    @staticmethod
    def _listify(*args):
        return [[arg] if isinstance(arg, str) else [a.strip() for a in arg] for arg in args]

    def _delete_weekends(self, df):

        if self._name == 'Bloomberg':
            # For Bloomberg, the index of the returned DataFrame is of type pd.Timestamp.
            assert isinstance(df.index[0], pd.Timestamp)
        elif self._name == 'Datastream':
            # For Datastream, the index of the returned DataFrame is string. So change it to pd.Timestamp.
            # It might get changed later using the index_type argument, but for now, it allows us to easily
            # remove the weekends.
            assert isinstance(df.index[0], str)
            df.index = [utils_dt.get_datetime(d, output_type=pd.Timestamp) for d in df.index]

        _num_rows = df.shape[0]
        df = df[~((df.index.weekday == 5) | (df.index.weekday == 6))]
        if _num_rows != df.shape[0]:
            print(str(_num_rows - df.shape[0]) + ' weekend days were deleted.')
        # Return the DataFrame.
        return df