import dateutil.parser
import datetime
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BMonthEnd
from numbers import Integral


def get_datetime(q_date, output_type=datetime.date, frmt='%Y-%m-%d'):
    """
    This function converts q_date into a different type. Allowed input and output types are str, datetime.datetime,
    datetime.date, pd.Timestamp and np.datetime64.
    :param q_date:          Date to be converted.
    :param output_type:     The desired output type.
    :param frmt:            Format when converting to str.
    :return:
    """
    # Treat q_date differently depending on its type.
    if type(q_date) == str:
        q_date = cvt_from_str(q_date, output_type)
    elif type(q_date) == datetime.datetime:
        q_date = cvt_from_datetime(q_date, output_type, frmt)
    elif type(q_date) == datetime.date:
        q_date = cvt_from_date(q_date, output_type, frmt)
    elif type(q_date) == pd.Timestamp:
        q_date = cvt_from_timestamp(q_date, output_type, frmt)
    elif type(q_date) == np.datetime64:
        q_date = cvt_from_datetime64(q_date, output_type, frmt)
    else:
        raise Exception('Cannot convert q_date of this type: ' + type(q_date).__name__ + '.')

    # Return q_date.
    return q_date


def cvt_from_str(q_date, output_type):

    if output_type == str:
        # Do nothing here.
        pass

    elif output_type == datetime.datetime:
        # Use the parser.
        q_date = dateutil.parser.parse(q_date)

    elif output_type == datetime.date:
        # Use the parser and the date() method.
        q_date = dateutil.parser.parse(q_date).date()

    elif output_type == pd.Timestamp:
        # Simply run the constructor.
        q_date = pd.Timestamp(q_date)

    elif output_type == np.datetime64:
        # Simply run the constructor.
        q_date = np.datetime64(q_date)

    # Return q_date.
    return q_date


def cvt_from_datetime(q_date, output_type, frmt):

    if output_type == str:
        q_date = q_date.strftime(frmt)

    elif output_type == datetime.datetime:
        # Do nothing here.
        pass

    elif output_type == datetime.date:
        # Use the date() method.
        q_date = q_date.date()

    elif output_type == pd.Timestamp:
        # Simply run the constructor.
        q_date = pd.Timestamp(q_date)

    elif output_type == np.datetime64:
        # Simply run the constructor.
        q_date = np.datetime64(q_date)

    # Return q_date.
    return q_date


def cvt_from_date(q_date, output_type, frmt):

    if output_type == str:
        q_date = q_date.strftime(frmt)

    elif output_type == datetime.datetime:
        q_date = datetime.datetime.combine(q_date, datetime.time())

    elif output_type == datetime.date:
        # Do nothing here.
        pass

    elif output_type == pd.Timestamp:
        # Simply run the constructor.
        q_date = pd.Timestamp(q_date)

    elif output_type == np.datetime64:
        # Simply run the constructor.
        q_date = np.datetime64(q_date)

    # Return q_date.
    return q_date


def cvt_from_timestamp(q_date, output_type, frmt):

    if output_type == str:
        q_date = q_date.strftime(frmt)

    elif output_type == datetime.datetime:
        q_date = q_date.to_pydatetime()

    elif output_type == datetime.date:
        q_date = q_date.to_pydatetime().date()

    elif output_type == pd.Timestamp:
        # Do nothing here.
        pass

    elif output_type == np.datetime64:
        q_date = q_date.to_datetime64()

    # Return q_date.
    return q_date


def cvt_from_datetime64(q_date, output_type, frmt):

    if output_type == np.datetime64:
        # Do nothing here.
        pass
    else:
        # Convert to pd.Timestamp and use _cvt_timestamp.
        q_date = cvt_from_timestamp(pd.Timestamp(q_date), output_type=output_type, frmt=frmt)

    # Return q_date.
    return q_date


def get_eo_month(q_date, months=0, output_type=datetime.date, beg_of_month=False, use_cal_days=False):

    # Assert that months is and integer (of some sort).
    assert isinstance(months, Integral)

    # Convert the input to a pd.Timestamp.
    q_date = get_datetime(q_date, output_type=pd.Timestamp)

    # Add to months in this case.
    if use_cal_days and not beg_of_month:
        months += 1

    # From this point on the type of q_date is pd.Timestamp.
    q_date += pd.DateOffset(months=months)

    #
    if use_cal_days:

        if beg_of_month:
            # Beginning of month
            output = q_date.replace(day=1)
        else:
            # End of month.
            q_date = q_date.replace(day=1)
            output = q_date - datetime.timedelta(days=1)

    else:

        # Initiate the Offset object.
        offset = BMonthEnd()
        #
        if beg_of_month:
            # Beginning of month. Get last business day of previous month then add one business day.
            output = offset.rollback(q_date) + pd.offsets.BDay(1)
        else:
            # End of month.
            output = offset.rollforward(q_date)

    # Return the output.
    return output if output_type is None else get_datetime(output, output_type=output_type)


def main():
    q_date = datetime.date.today()
    months = 4
    print(get_eo_month(q_date, months=months), get_eo_month(q_date, months=months, use_cal_days=True))
    print(get_eo_month(q_date, months=months, beg_of_month=True), get_eo_month(q_date, months=months, use_cal_days=True, beg_of_month=True))


if __name__ == '__main__':
    main()
