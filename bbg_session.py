import blpapi
from optparse import OptionParser
from tools import utils_df
from tools import utils
import datetime
import pandas as pd
import numpy as np
from data_provider_base import DataProviderBase
from dateutil.parser import parse

"""
Useful link:
https://github.com/691175002/BLPInterface/blob/master/blp.py
"""


def get_session():

    try:
        bbg = BloombergSession()
    except:
        bbg = None
    # Return the session.
    return bbg


class RequestError(Exception):
    """A RequestError is raised when there is a problem with a Bloomberg API response."""

    def __init__(self, value, description):
        self.value = value
        self.description = description
        print('[ERROR]')
        print(value)


class BloombergSession(DataProviderBase):

    def __init__(self):

        # Call the base class constructor.
        DataProviderBase.__init__(self, 'Bloomberg')

        # Define date format for this sub-class.
        self._format = '%Y%m%d'

        _hst_data = ('PX_LAST', 'OPEN', 'HIGH', 'LOW', 'FREE_FLOAT_MARKET_CAP')
        _ref_data = ('DUR_ADJ_MID', 'INFLATION_ADJ_DUR_MID')

        self._FIELD_MAP = dict(zip(_hst_data, ['HISTORICAL_DATA']*len(_hst_data)))
        self._FIELD_MAP.update(dict(zip(_ref_data, ['REFERENCE_DATA']*len(_ref_data))))

        # Update the keyword map.
        self._kwrd_map.update({'P': 'periodicitySelection', 'PER': 'periodicitySelection',
                               'PERIOD': 'periodicitySelection', 'F': 'periodicitySelection',
                               'FREQ': 'periodicitySelection', 'FREQUENCY': 'periodicitySelection',
                               'PERIODICITYSELECTION': 'periodicitySelection'})

        self._kval_map = {'periodicitySelection': {'D': 'DAILY', 'DAY': 'DAILY', 'DAILY': 'DAILY',
                                                   'M': 'MONTHLY', 'MONTH': 'MONTHLY', 'MONTHLY': 'MONTHLY',
                                                   'Q': 'QUARTERLY', 'QUARTER': 'QUARTERLY', 'QUARTERLY': 'QUARTERLY',
                                                   'Y': 'YEARLY', 'YEAR': 'YEARLY', 'YEARLY': 'YEARLY'}
                          }

        self._ref_data_service = None
        self._session = None
        self._err_msg = None

        self._construct()

        if self._err_msg is not None:
            raise Exception('Could not initiate the Bloomberg Session object: ' + self._err_msg)

        self._defaults = {
            'nonTradingDayFillOption': 'NON_TRADING_WEEKDAYS',  # 'Days', Default
            'nonTradingDayFillMethod': 'PREVIOUS_VALUE',  # 'Fill', Default
            'periodicityAdjustment': 'CALENDAR',  # 'Per',  Default
            'periodicitySelection': 'DAILY',  # 'Per',  Default
            'adjustmentNormal': False,
            'adjustmentAbnormal': False,
            'adjustmentSplit': True,
            'adjustmentFollowDPDF': False
        }

    def _construct(self):

        try:
            parser = OptionParser(description="Retrieve reference data.")
            parser.add_option("-a",
                              "--ip",
                              dest="host",
                              help="server name or IP (default: %default)",
                              metavar="ipAddress",
                              default="localhost")
            parser.add_option("-p",
                              dest="port",
                              type="int",
                              help="server port (default: %default)",
                              metavar="tcpPort",
                              default=8194)

            (options, args) = parser.parse_args()

            # Fill SessionOptions
            session_options = blpapi.SessionOptions()
            session_options.setServerHost(options.host)
            session_options.setServerPort(options.port)

            # Create a Session
            print("Connecting to %s:%s" % (options.host, options.port))
            self._session = blpapi.Session(session_options)

        except Exception as err:
            self._err_msg = 'Failed to initiate session.'
            return

            # Start a Session
        if not self._session.start():
            self._err_msg = 'Failed to start session.'
            return

        #  Open service to get historical data from
        if not self._session.openService("//blp/refdata"):
            self._err_msg = 'Failed to open //blp/refdata'
            return

        self._ref_data_service = self._session.getService("//blp/refdata")

    def get_historical(self, securities=None, fields=None, start_date='19900101', end_date=None, index_type=None,
                       batch_size=None, info_on_screen=False, screen_msg=None, delete_weekends=True, **kwargs):


        # Do some (common) things before proceeding. This includes checking the kwargs and building a dictionary with
        # methods to operate on the final DataFrame.
        kwargs = self._do_pre(kwargs)

        # Unpack some data.
        securities, fields, start_date, end_date, kwargs = self._unpack(securities, fields, start_date, end_date,
                                                                        kwargs)
        # Set some defaults.
        self._set_defaults(start_date, end_date, **kwargs)

        def h_get_data(secs):
            msgs = self._send_request('HistoricalData', secs, fields, self._defaults)
            return self._get_historical(msgs, fields)

        # Get the data.
        df = self._get_data(h_get_data, securities, batch_size, info_on_screen=info_on_screen, screen_msg=screen_msg)

        # Do some post stuff before exiting. This includes changing the type of the index, and running any custom
        # methods on df.
        harmonize = kwargs.pop('harmonize', False)
        df = self._do_post(df, index_type, harmonize=harmonize, delete_weekends=delete_weekends)

        # Return the DataFrame.
        return df

    def get_reference(self, securities, fields, **kwargs):

        securities, fields = self._listify(securities, fields)

        # Get a unique list of securities.
        securities = utils.get_unique_list(securities)

        msgs = self._send_request('ReferenceData', securities, fields, elements=kwargs)
          
        if len(msgs) == 1:
            df_data = self._get_reference(fields, msgs)
        else:
            df_temp = [None] * len(msgs)
            for i, msg in enumerate(msgs):
                df_temp[i] = self._get_reference(fields, [msgs[i]])
            df_data = pd.concat(df_temp, axis=0)
        return df_data

    def get_point(self, security, field, overrides, _date):

        if field not in self._FIELD_MAP:
            raise Exception('This field is not recognised. Please add to map in bbg_session.py')

        if self._FIELD_MAP[field] == 'HISTORICAL_DATA':
            show_date = _date
            df_temp = self.get_historical(securities=security, fields=field, start_date=_date, end_date=_date,
                                          index_type=datetime.date, **overrides)
            value = df_temp.loc[_date, [security, field]].squeeze()

        elif self._FIELD_MAP[field] == 'REFERENCE_DATA':
            show_date = None
            df_temp = self.get_reference(security, field)
            value = df_temp.loc[security, field]

        else:
            raise Exception('This field is not recognised. Please add to map in bbg_session.py')

        return value, show_date

    def _get_reference(self, fields, msgs, is_bulk=False):
        # Assert that there is only one message.
        assert len(msgs) == 1
        # Unpack the message.
        msg = msgs[0]
        # Define output columns.
        columns = ['security'] + fields

        # Unpack the security data list.
        security_data_list = [msg.getElement('securityData').getValueAsElement(i) for i in
                              range(msg.getElement('securityData').numValues())]

        # Get the data. Note that for each sec_dta, the data is constructed by adding the security name to the field
        # data.
        if is_bulk:
            data = [self._h_get_data(sec_dta.getElement('fieldData'), fields, is_bulk=is_bulk) for sec_dta in security_data_list][0] 
            _test_el = security_data_list[0].getElement('fieldData').getElement(fields[0]).getValue(0)
            columns = [str(_test_el.getElement(i).name()) for i in range(_test_el.numElements())]
            index_name = columns[0]
        else:
            data = [[sec_dta.getElement('security').getValue()] + self._h_get_data(sec_dta.getElement('fieldData'), fields,
                    is_bulk=is_bulk) for sec_dta in security_data_list]
            index_name = None

        # Construct the DataFrame.
        df_data = pd.DataFrame(data=data, columns=columns)

        if df_data.shape[1] > 1:
            df_data.set_index(columns[0], drop=True, inplace=True)
            df_data.index.name = index_name

        # Return the DataFrame.
        return df_data

    def _get_historical(self, msgs, fields):

        df_data = []
        keys = []
        columns = ['date'] + fields

        # Iterate over messages.
        for msg in msgs:

            # Get securityData and fieldData from the message (msg).
            security_data = msg.getElement('securityData')
            field_data = security_data.getElement('fieldData')

            # Unpack the field_data using list comprehension.
            field_data_list = [field_data.getValueAsElement(i) for i in range(field_data.numValues())]


            data = [self._h_get_data(fld, columns) for fld in field_data_list]

            # If there is data, stick it in a DataFrame (df_temp) and append the DataFrame to a list (df_data).
            if len(data) > 0:
                # If there is data, then setup a DataFrame.
                df_temp = pd.DataFrame(data=data, columns=columns)
                # Set the Date-column as index.
                df_temp.set_index(columns[0], drop=True, inplace=True)
                # Change type of the index to datetime.datetime.
                df_temp.index = pd.to_datetime(df_temp.index)
                # Replace any missing history with nans.
                df_temp.replace('#N/A History', np.nan, inplace=True)
                # Append the security name to the valid keys.
                keys.append(security_data.getElementAsString('security'))
                # Append the DataFrame to the list df_data.
                df_data.append(df_temp)

        # Create a DataFrame for the total output. If df_data has no elements, then return an empty DataFrame with the
        # correct column names. If df_data is a valid list, then concatenate the individual elements into one DataFrame.
        if len(df_data) == 0:
            df = pd.DataFrame(columns=columns)
        else:
            # Create a single DataFrame from a list containing separate DataFrames.
            df = utils_df.concat_dfs_from_list(df_data, keys=keys, names=['Security', 'Field'])
            # Set the name of the index to be None.
            df.index.name = None

        # Return the DataFrame.
        return df

    def _send_request(self, request_type, securities, fields, elements={}):

        request = self._ref_data_service.createRequest(request_type + 'Request')

        # Append multiple securities (e.g. 'SPX Index') to request
        for s in securities:
            request.getElement("securities").appendValue(s)

        # Append multiple fields (e.g. PX_LAST) to request
        for f in fields:
            request.getElement("fields").appendValue(f)

        for k, v in elements.items():
            if hasattr(v, 'strftime'):
                v = v.strftime(self._format)
            # Set value for each request parameter (e.g. 'periodicitySelection', 'DAILY')
            try:
                request.set(k, v)
            except Exception as err:
                print('Unable to set element with name ' + k + ' and value ' + str(v) + '.')
                print(str(err))
                raise err

        self._session.sendRequest(request)

        response = self._get_response(request_type)

        return response

    def _get_response(self, request_type):
        response = []
        while True:
            event = self._session.nextEvent(100)
            for msg in event:
                if msg.hasElement('responseError'):
                    raise RequestError(msg.getElement('responseError'), 'Response Error')

                if msg.hasElement('securityData'):

                    if msg.getElement('securityData').hasElement('fieldExceptions') \
                            and (msg.getElement('securityData').getElement('fieldExceptions').numValues() > 0):
                        raise RequestError(msg.getElement('securityData').getElement('fieldExceptions'),
                                           'Field Error')

                    if msg.getElement('securityData').hasElement('securityError'):
                        raise RequestError(msg.getElement('securityData').getElement('securityError'), 'Security Error')

                if msg.messageType() == request_type + 'Response':
                    response.append(msg)

            if event.eventType() == blpapi.Event.RESPONSE:
                break

        return response

    def _set_defaults(self, start_date, end_date, **kwargs):


        if start_date is not None:
            self._defaults['startDate'] = start_date

        if end_date is not None:
            self._defaults['endDate'] = end_date



        # Add new keys to defaults.
        self._defaults.update(kwargs)

    @staticmethod
    def _h_get_data(fld, field_names, is_bulk=False):
        _num_els = fld.numElements()
        _num_fld_nms = len(field_names)
        if _num_els == _num_fld_nms:
            if is_bulk:
                # Can only do one field in bulk.
                assert len(field_names) == 1
                sub_field = fld.getElement(field_names[0])
                return [[it.getElementValue(i) for i in range(it.numElements())] for it in sub_field.values()]

            else:
                # If the number of elements in fld corresponds to num_elmnts, then a straight forward unpack can be used
                # under the assumption that the elements returned are in the same order as the requested fields.
                return [fld.getElement(i).getValue() for i in range(fld.numElements())]

        else:

            output = [np.nan] * _num_fld_nms
            # Now iterate over the elements.
            for i in range(_num_els):
                _name = str(fld.getElement(i).name())
                _idx = field_names.index(_name)
                output[_idx] = fld.getElementValue(i)
            #
            return output

    def _unpack(self, securities, fields, start_date, end_date, kwargs):
        # End Date.
        if end_date is None:
            end_date = kwargs.pop('end_date', (datetime.date.today() - pd.offsets.BDay(1)).strftime(self._format))
        # Listify the securities and fields.
        securities, fields = self._listify(securities, fields)
        # Start Date.
        harmonize = kwargs.pop('harmonize', True)
        # Start date.
        if start_date is None:
            start_date = self._def_start_date
        if start_date is None:
            start_date = self._get_start_date(securities, start_date, harmonize=harmonize)
        # Return the inputs.
        return securities, fields, start_date, end_date, kwargs

    def _get_start_date(self, securities, start_date, harmonize=True):


        # Get possible start_dates.
        _start_dates = self._get_possible_start_dates(securities)

        # The greatest value across all securities for the two fields is chosen as the default start date, and also used
        # if the passed start_date is before the date when the historic data begins.
        if harmonize:


            _start_date = max(_start_dates) if len(_start_dates) > 0 else None

        else:


            _start_date = min(_start_dates) if len(_start_dates) > 0 else None

        if start_date is None:
            # No start_date has been passed, so use _start_date.
            start_date = _start_date
        else:
            if _start_date is None:
                # Do nothing here.
                pass
            else:
                # A start_date has been passed, so get it in the right format.
                _comp_date = self._get_comp_date(start_date)
                # Now get the maximum of _start_date and _comp_date.
                start_date = max(_start_date, _comp_date)

        # Return the start date.
        return start_date

    @staticmethod
    def _parse_bbg_string(value):
        value = value.upper()
        for x in ('DAILY', 'WEEKLY', 'MONTHLY', 'QUARTERLY', 'YEARLY'):
            if x in value:
                value = value.replace(x, '').strip()
                break
        value = parse(value).date()
        return value

    def _get_possible_start_dates(self, securities):
        # Fields to obtain start dates.
        fields = ['HISTORY_START_DT', 'INDX_HIST_START_DT_MONTHLY']
        # Call Bloomberg with a Reference data request for the two static fields.
        msgs = self._send_request('ReferenceData', securities, fields)
        df_start_dates = self._get_reference(fields, msgs)
        # Stick the values of the DataFrame in single list.
        start_dates = np.reshape(df_start_dates.values, (1, np.prod(df_start_dates.shape))).tolist()[0]
        # Remove any nans. Possibly values are now either strings or datetimes.
        start_dates = [x for x in start_dates if isinstance(x, (str, datetime.datetime, datetime.date))]
        # Parse any strings.
        start_dates = [self._parse_bbg_string(x) if isinstance(x, str) else x for x in start_dates]
        # Return the star_dates.
        return start_dates

    @staticmethod
    def _get_comp_date(start_date):
        if start_date is None:
            _comp_date = None
        else:
            # If a start_date was passed we take the greater value of start_date and _start_date.
            if isinstance(start_date, str):
                # If start_date was passed as a string it must be parsed.
                _comp_date = parse(start_date).date()
            else:
                # Convert start_date to type datetime.date.
                _comp_date = start_date.date() if hasattr(start_date, 'date') else start_date
        return _comp_date

    def get_reference_or(self, securities, fields, overrides, is_bulk=False):
        # https://stackoverflow.com/questions/44720573/how-to-make-this-call-using-bloomberg-api

        securities, fields = self._listify(securities, fields)

        # Get a unique list of securities.
        securities = utils.get_unique_list(securities)


        request = self._ref_data_service.createRequest('ReferenceDataRequest')

        # Append multiple securities (e.g. 'SPX Index') to request
        for s in securities:
             request.getElement("securities").appendValue(s)
        #request.getElement("securities").appendValue(securities)  # 3M Curncy

        # Append multiple fields (e.g. PX_LAST) to request
        for f in fields:
            request.getElement("fields").appendValue(f)

        for k, v in overrides.items():
            overrides = request.getElement('overrides').appendElement()
            overrides.setElement('fieldId', k)
            overrides.setElement('value', v)
 
        self._session.sendRequest(request)

        msgs = self._get_response('ReferenceData')

        if len(msgs) == 1:
            df_data = self._get_reference(fields, msgs, is_bulk=is_bulk)
        else:
            df_temp = [None] * len(msgs)
            for i, msg in enumerate(msgs):
                df_temp[i] = self._get_reference(fields, [msgs[i]], is_bulk=is_bulk)
            df_data = pd.concat(df_temp, axis=0)
        return df_data
        



