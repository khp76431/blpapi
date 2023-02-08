from bbg_session import BloombergSession
from dateutil.parser import parse
import pandas as pd




# Initiate a Bloomberg session.
bbg = BloombergSession()

securities = ['ES1 A:00_0_R? Index']
fields = ['PX_LAST']
start_date = pd.to_datetime('01/01/1990')  # parse('20170101')
end_date = pd.to_datetime('today')# parse('20180418')
harmonize = False

# These are additional functions which will be applied before returning the DataFrame.
h_fun = pd.DataFrame.fillna
args = {'method': 'ffill', 'inplace': True}
_h_fun = None  # ((h_fun, args), (lambda x: x, {}))

df = bbg.get_historical(securities=securities,  fields=fields, start_date=start_date, end_date=end_date,
                        harmonize=harmonize, period='D', h_fun=_h_fun)

print(df)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    