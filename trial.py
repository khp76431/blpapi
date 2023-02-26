# -*- coding: utf-8 -*-

from data_connect.bbg_session import BloombergSession
import pandas as pd
import numpy as np
import utilities as util
from scipy.optimize import minimize_scalar
from collections import deque
import matplotlib.pyplot as plt

# start_date        = pd.to_datetime('01/01/1990')                                   
# df_idx_high_daily = util.get_bbg_data(df_config_main['RISK_TICKER'].values , ['PX_HIGH'], start_date, pd.to_datetime('today'), df_config_main , 'RISK_TICKER', 'D')
# df_idx_low_daily  = util.get_bbg_data(df_config_main['RISK_TICKER'].values , ['PX_LOW'], start_date, pd.to_datetime('today'), df_config_main , 'RISK_TICKER', 'D')
# df_idx_volume     = util.get_bbg_data(df_config_main['RISK_TICKER'].values , ['PX_VOLUME'], start_date, pd.to_datetime('today'), df_config_main , 'RISK_TICKER', 'D')
# df_prices_now     = util.get_bbg_data(df_config_main['RISK_TICKER'].values , ['PX_LAST'], start_date, pd.to_datetime('today'), df_config_main , 'RISK_TICKER', 'D')

import numpy as np
import matplotlib.pyplot as plt

def transformed_sigmoid(x , k):
    return 2 * (1 / (1 + np.exp(-x*k))) - 1

x = np.linspace(-1, 1, 100)
y_k1 = transformed_sigmoid(x*10 , k = 1)
y_k5 = transformed_sigmoid(x*10 , k = 5)
y_k05 = transformed_sigmoid(x*10 , k = 0.5)


plt.plot(x, x, label='Linear')
plt.plot(x, y_k1, label='k=1')
plt.plot(x, y_k5, label='k=5')
plt.plot(x, y_k05, label='k=0.5')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Sigmoid Function with Adjustable Convexity")
plt.legend()
plt.grid()
plt.show()



def moving_average(df_prices, period):
    # Calculate the moving average
    ma = df_prices.rolling(window=period).mean()
    # Add the moving average to the DataFrame
    
    df_signal = pd.DataFrame(data= 0, index  = df_prices.index, columns = df_prices.columns)
    df_signal[df_prices > ma] = 1
    df_signal[df_prices < ma] = -1
    
    
    return ma , df_signal

#df_ma_prices , df_signal = moving_average(df_prices_now, 200)


def relative_strength_index(df_prices, period):
    # Calculate the change in price
    delta       = df_prices.diff()

    # Create the gain and loss columns
    gain        = delta.where(delta > 0, 0)
    loss        = -delta.where(delta < 0, 0)

    # Calculate the average gain and loss
    avg_gain    = gain.rolling(window=period).mean()
    avg_loss    = loss.rolling(window=period).mean()

    # Calculate the relative strength
    rs          = avg_gain / avg_loss

    # Calculate the relative strength index
    df_RSI      = 100 - (100 / (1 + rs))

    df_signal = pd.DataFrame(data= 0, index  = df_prices.index, columns = df_prices.columns)
    df_signal[df_RSI > 70] = -1
    df_signal[df_RSI < 30] = 1

    return df_RSI , df_signal

#df_RSI , df_signal = relative_strength_index(df_prices_now, 14)


# Stochastic Oscillator
def stochastic_oscillator(df_prices, period_k, period_d):
    # Calculate the high, low and close over the period
    low = df_prices.rolling(window=period_k).min()
    high = df_prices.rolling(window=period_k).max()
    close = df_prices

    # Calculate the stochastic oscillator %K
    stoch_k = 100 * (close - low) / (high - low)

    # Calculate the stochastic oscillator %D
    stoch_d = stoch_k.rolling(window=period_d).mean()

    return stoch_k , stoch_d

#stoch_k , stoch_d = stochastic_oscillator(df_prices_now, 14, 3)



def macd(df_prices, period_fast, period_slow):
    # Calculate the moving averages
    ma_fast = df_prices.rolling(window=period_fast).mean()
    ma_slow = df_prices.rolling(window=period_slow).mean()

    # Calculate the MACD
    macd = ma_fast - ma_slow

    # Calculate the signal line
    signal = macd.rolling(window=9).mean()

    # Calculate the histogram
    hist = macd - signal


    return macd , signal , hist

#macd , signal , hist = macd(df_prices_now, 12, 26)


def ichimoku_kinko_hyo(df_prices , df_idx_high_daily ,df_idx_low_daily , period1, period2, period3):
    """ ichimoku_kinko_hyo takes in four parameters: the DataFrame of prices (df_prices), the period for the conversion line (Tenkan-sen) (period1), the period for the base line (Kijun-sen) (period2)
    , and the period for the leading span B (Senkou Span B) (period3). The function first calculates the conversion line (Tenkan-sen) by taking the average of the high and low prices over the period1 window. 
    Next, it calculates the base line (Kijun-sen) by taking the average of the high and low prices over the period2 window. Then it calculates the leading span A (Senkou Span A) by taking the average of the 
    Tenkan-sen and Kijun-sen lines and shifting it forward by period2 periods. After that, it calculates the leading span B (Senkou Span B) by taking the average of the highest high and lowest low prices over 
    the period3 window and shifting it forward by period2 periods. Finally, it calculates the lagging span (Chikou Span) by shifting the close prices backwards by period2 periods"""
    # Calculate the conversion line (Tenkan-sen)
    tenkan_sen = (df_idx_high_daily.rolling(window=period1).mean() + df_idx_low_daily.rolling(window=period1).mean()) / 2
    
    # Calculate the base line (Kijun-sen)
    kijun_sen = (df_idx_high_daily.rolling(window=period2).mean() + df_idx_low_daily.rolling(window=period2).mean()) / 2
    
    # Calculate the leading span A (Senkou Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(period2)
    
    # Calculate the leading span B (Senkou Span B)
    senkou_span_b = ((df_idx_high_daily.rolling(window=period3).max() + df_idx_low_daily.rolling(window=period3).min()) / 2).shift(period2)
    
    # Calculate the lagging span (Chikou Span)
    chikou_span = df_prices.shift(-period2)
    

    return tenkan_sen , kijun_sen , senkou_span_a , senkou_span_b , chikou_span
    


#tenkan_sen , kijun_sen , senkou_span_a , senkou_span_b , chikou_span = ichimoku_kinko_hyo(df_prices_now, df_idx_high_daily ,df_idx_low_daily, 9, 26, 52)



    
def get_adx(high, low, close, lookback):
    ### https://medium.com/codex/algorithmic-trading-with-average-directional-index-in-python-2b5a20ecf06a
    plus_dm                 = high.diff()
    minus_dm                = low.diff()
    plus_dm[plus_dm < 0]    = 0
    minus_dm[minus_dm > 0]  = 0
    
    tr1                     = pd.DataFrame(high - low)
    tr2                     = pd.DataFrame(abs(high - close.shift(1)))
    tr3                     = pd.DataFrame(abs(low - close.shift(1)))
    frames                  = [tr1, tr2, tr3]
    tr                      = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
    atr                     = tr.rolling(lookback).mean()
    
    plus_di                 = 100 * (plus_dm.ewm(alpha = 1/lookback).mean() / atr)
    minus_di                = abs(100 * (minus_dm.ewm(alpha = 1/lookback).mean() / atr))
    dx                      = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx                     = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
    adx_smooth              = adx.ewm(alpha = 1/lookback).mean()

    df                      = pd.DataFrame(data = np.nan, index = df_prices_now.index, columns = ['+DI','-DI','ADX'])
    df['+DI']               = plus_di
    df['-DI']               = minus_di
    df['ADX']               = adx_smooth

    return df

def plot_ADX(df_price , df_ADX , df_POS , asset, s_date, e_date):
    
    plt.style.use('fivethirtyeight')
    plt.rcParams['figure.figsize'] = (20,10)    
    ax1         = plt.subplot2grid((11,1), (0,0), rowspan = 5, colspan = 1)
    ax2         = plt.subplot2grid((11,1), (6,0), rowspan = 5, colspan = 1)
    ax1.plot(df_price[asset].loc[s_date:e_date], linewidth = 2, color = '#ff9800')
    ax1.set_title('CLOSING PRICE: '+asset)

    ax1.plot(df_price[asset].loc[s_date:e_date].index, df_POS[asset]['BUY PRICE'].loc[s_date:e_date], marker = '^', color = '#26a69a', markersize = 14, linewidth = 0, label = 'BUY SIGNAL')
    ax1.plot(df_price[asset].loc[s_date:e_date].index, df_POS[asset]['SELL PRICE'].loc[s_date:e_date], marker = 'v', color = '#f44336', markersize = 14, linewidth = 0, label = 'SELL SIGNAL')

    ax2.plot(df_ADX[asset]['+DI'].loc[s_date:e_date], color = '#26a69a', label = '+ DI 14', linewidth = 3, alpha = 0.3)
    ax2.plot(df_ADX[asset]['-DI'].loc[s_date:e_date], color = '#f44336', label = '- DI 14', linewidth = 3, alpha = 0.3)
    ax2.plot(df_ADX[asset]['ADX'].loc[s_date:e_date], color = '#2196f3', label = 'ADX 14', linewidth = 3)
    ax2.axhline(25, color = 'grey', linewidth = 2, linestyle = '--')
    ax2.legend()
    ax2.set_title('ADX 14: '+asset)
    plt.show()


def implement_adx_strategy(prices, pdi, ndi, adx):
    
    buy_price   = []
    sell_price  = []
    adx_signal  = []
    signal      = 0
    
    for i in range(len(prices)):
        if adx[i-1] < 25 and adx[i] > 25 and pdi[i] > ndi[i]:
            if signal != 1:
                buy_price.append(prices[i])
                sell_price.append(np.nan)
                signal = 1
                adx_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                adx_signal.append(0)
        elif adx[i-1] < 25 and adx[i] > 25 and ndi[i] > pdi[i]:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(prices[i])
                signal = -1
                adx_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                adx_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            adx_signal.append(0)
            
    return buy_price, sell_price, adx_signal



def implement_adx_strategy(df_prices_now, df_ADX , asset):
    
    prices                  = df_prices_now[asset]
    pdi                     = df_ADX[asset]['+DI']
    ndi                     = df_ADX[asset]['-DI']
    adx                     = df_ADX[asset]['ADX']
    
    df                      = pd.DataFrame(data = np.nan, index = df_prices_now.index, columns = ['BUY PRICE', 'SELL PRICE', 'ADX SIGNAL'] )
    df['ADX SIGNAL']        = 0
    prev_signal             = 0
    signal                  = 0

    for i in range(len(prices)):
        # if ADX rises from below 354 to above 25 and +DI now > -DI
        if adx.iloc[i-1] < 25 and adx.iloc[i] > 25 and pdi.iloc[i] > ndi.iloc[i]:
            if signal != 1:
                prev_signal                 = signal
                df['BUY PRICE'].iloc[i]     = prices.iloc[i]
                df['SELL PRICE'].iloc[i]    = np.nan
                signal                      = 1
                df['ADX SIGNAL'].iloc[i]    = signal
        # if ADX fallls from above 25 to below 25 and -DI now < +DI                
        elif adx.iloc[i-1] < 25 and adx.iloc[i] > 25 and ndi.iloc[i] > pdi.iloc[i]:
            if signal != -1:
                prev_signal                 = signal
                df['BUY PRICE'].iloc[i]     = np.nan
                df['SELL PRICE'].iloc[i]    = prices.iloc[i]
                signal                      = -1
                df['ADX SIGNAL'].iloc[i]    = signal
            
    return df

# df_ADX                      = defaultdict()
# df_POS                      = defaultdict()

# for asset in df_prices_now.columns:
#     df_ADX[asset] = get_adx(df_idx_high_daily[asset], df_idx_low_daily[asset], df_prices_now[asset], 14)
#     df_POS[asset] = implement_adx_strategy(df_prices_now, df_ADX , asset)

# plot_ADX(df_prices_now , df_ADX , df_POS, 'USA FUTURE', '2020', '2023')





def money_flow_index(high, low, close, volume, n):
    """
    Compute the Money Flow Index (MFI) for a given DataFrame and time period (n)
    https://randerson112358.medium.com/algorithmic-trading-strategy-using-money-flow-index-mfi-python-aa46461a5ea5
    """
    df                      = pd.DataFrame(data = np.nan, index = close.index, columns = ['High','Low','Close','Volume' , 'TP','MF','Positive MF','Negative MF' , 'Positive MF Sum' , 'Negative MF Sum', 'MFR', 'MFI', 'SIGNAL'])
    df['High']              = high
    df['Low']               = low
    df['Close']             = close
    df['Volume']            = volume
    # Create new columns for typical price and money flow
    df['TP']                = (df['High'] + df['Low'] + df['Close']) / 3
    df['MF']                = df['TP'] * df['Volume']
    
    # Initialize variables for positive and negative money flow
    pos_mf                  = []
    neg_mf                  = []
    
    # Loop through the DataFrame to calculate positive and negative money flow
    for i in range(1, len(df)):
        if df.iloc[i]['TP'] > df.iloc[i-1]['TP']:
            df['Positive MF'].iloc[i] = df.iloc[i]['MF']
            df['Negative MF'].iloc[i] = 0
        else:
            df['Positive MF'].iloc[i] = 0
            df['Negative MF'].iloc[i] = df.iloc[i]['MF']
    
    # Calculate the n-period positive and negative money flow sum
    df['Positive MF Sum']   = df['Positive MF'].rolling(window=n).sum()
    df['Negative MF Sum']   = df['Negative MF'].rolling(window=n).sum()
    
    # Calculate the money flow ratio and the money flow index
    df['MFR']               = df['Positive MF Sum'] / df['Negative MF Sum']
    df['MFI']               = 100 - (100 / (1 + df['MFR']))
    
    df['SIGNAL']                    =  0
    df['SIGNAL'][df['MFI'] > 80]   = -1
    df['SIGNAL'][df['MFI'] < 20]   =  1
    
    return df 

def plot_MFI(mfi , asset):
    plt.style.use('fivethirtyeight')
    df2         = pd.DataFrame()
    df2['MFI']  = mfi['MFI']
    #Create and plot the graph
    plt.figure(figsize=(12.2,4.5)) #width = 12.2in, height = 4.5
    plt.plot( df2['MFI'],  label='MFI '+asset)#plt.plot( X-Axis , Y-Axis, line_width, alpha_for_blending,  label)
    plt.axhline(10, linestyle='--', color = 'orange')  #Over Sold line (Buy)
    plt.axhline(20, linestyle='--',color = 'blue')  #Over Sold Line (Buy)
    plt.axhline(80, linestyle='--', color = 'blue')  #Over Bought line (Sell)
    plt.axhline(90, linestyle='--', color = 'orange')  #Over Bought line (Sell)
    plt.title('MFI: '+asset)
    plt.ylabel('MFI Values',fontsize=18)
    plt.legend(df2.columns.values, loc='upper left')
    plt.show()


# df_MFI   = defaultdict()

# for asset in df_prices_now.columns:
#     df_MFI[asset] = money_flow_index(df_idx_high_daily[asset], df_idx_low_daily[asset], df_prices_now[asset]  , df_idx_volume[asset], 14)

# plot_MFI(df_MFI['USA FUTURE'].loc['2022'] , 'USA FUTURE')




def compute_obv(df):
    # Create a new column to hold the OBV values
    df['OBV'] = 0

    # Iterate through the dataframe and calculate the OBV values
    for i in range(1, len(df)):
        if df['Close'][i] > df['Close'][i-1]:
            df['OBV'][i] = df['OBV'][i-1] + df['Volume'][i]
        elif df['Close'][i] < df['Close'][i-1]:
            df['OBV'][i] = df['OBV'][i-1] - df['Volume'][i]
        else:
            df['OBV'][i] = df['OBV'][i-1]

    return df

# # Example usage
# df = pd.read_csv('stock_data.csv')
# df = compute_obv(df)
# print(df)






def nadarya_watson_indicator(df, high_col, low_col, alpha=0.95):
    """
    Computes the Nadarya-Watson envelope technical indicator for a given dataframe and column names.
    :param df: Dataframe containing the high and low data.
    :param high_col: Column name for the high data.
    :param low_col: Column name for the low data.
    :param alpha: Significance level for the envelope.
    :return: Dataframe containing the high, low, and envelope data.
    """
    # Compute average and standard deviation
    average = (df[high_col] + df[low_col])/2
    std = (df[high_col] - df[low_col])/2
    
    # Define the objective function for the optimization
    def obj_func(s):
        df['average_hat'] = average * s
        df['residual'] = average - df['average_hat']
        n = len(df)
        return (1 - alpha) * (n - 2) / n * df['residual'].std()
    
    # Find the slope of the envelope using scalar minimization
    res = minimize_scalar(obj_func)
    s = res.x
    
    # Compute the upper and lower bounds of the envelope
    df['upper_envelope'] = average * s + std
    df['lower_envelope'] = average * s - std
    
    return df

# # Example usage
# df = pd.read_csv("financial_data.csv")
# envelope_df = nadarya_watson_indicator(df, 'high', 'low')



#https://raposa.trade/blog/the-complete-guide-to-calculating-the-parabolic-sar-in-python/
class PSAR:

    def __init__(self, init_af=0.02, max_af=0.2, af_step=0.02):
        self.max_af             = max_af
        self.init_af            = init_af
        self.af                 = init_af
        self.af_step            = af_step
        self.extreme_point      = None
        self.high_price_trend   = []
        self.low_price_trend    = []
        self.high_price_window  = deque(maxlen=2)
        self.low_price_window   = deque(maxlen=2)
        
        # Lists to track results
        self.psar_list          = []
        self.af_list            = []
        self.ep_list            = []
        self.high_list          = []
        self.low_list           = []
        self.trend_list         = []
        self._num_days          = 0

    def calcPSAR(self, high, low):
        if self._num_days >= 3:
            psar = self._calcPSAR()
        else:
            psar = self._initPSARVals(high, low)

        psar = self._updateCurrentVals(psar, high, low)
        self._num_days += 1

        return psar

    def _initPSARVals(self, high, low):
        if len(self.low_price_window) <= 1:
            self.trend = None
            self.extreme_point = high
            return None
    
        if self.high_price_window[0] < self.high_price_window[1]:
            self.trend = 1
            psar = min(self.low_price_window)
            self.extreme_point = max(self.high_price_window)
        else: 
            self.trend = 0
            psar = max(self.high_price_window)
            self.extreme_point = min(self.low_price_window)
      
        return psar

    def _calcPSAR(self):
        prev_psar = self.psar_list[-1]
        if self.trend == 1: # Up
            psar = prev_psar + self.af * (self.extreme_point - prev_psar)
            psar = min(psar, min(self.low_price_window))
        else:
            psar = prev_psar - self.af * (prev_psar - self.extreme_point)
            psar = max(psar, max(self.high_price_window))
    
        return psar

    def _updateCurrentVals(self, psar, high, low):
        if self.trend == 1:
            self.high_price_trend.append(high)
        elif self.trend == 0:
            self.low_price_trend.append(low)
      
        psar = self._trendReversal(psar, high, low)
      
        self.psar_list.append(psar)
        self.af_list.append(self.af)
        self.ep_list.append(self.extreme_point)
        self.high_list.append(high)
        self.low_list.append(low)
        self.high_price_window.append(high)
        self.low_price_window.append(low)
        self.trend_list.append(self.trend)
    
        return psar

    def _trendReversal(self, psar, high, low):
        # Checks for reversals
        reversal = False
        if self.trend == 1 and psar > low:
            self.trend = 0
            psar = max(self.high_price_trend)
            self.extreme_point = low
            reversal = True
        elif self.trend == 0 and psar < high:
            self.trend = 1
            psar = min(self.low_price_trend)
            self.extreme_point = high
            reversal = True
        
        if reversal:
            self.af = self.init_af
            self.high_price_trend.clear()
            self.low_price_trend.clear()
        else:
            if high > self.extreme_point and self.trend == 1:
              self.af = min(self.af + self.af_step, self.max_af)
              self.extreme_point = high
            elif low < self.extreme_point and self.trend == 0:
              self.af = min(self.af + self.af_step, self.max_af)
              self.extreme_point = low
          
        return psar



def plot_SAR(data , asset):
    colors      = plt.rcParams['axes.prop_cycle'].by_key()['color']
    psar_bull   = data.loc[data['Trend']==1]['PSAR']
    psar_bear   = data.loc[data['Trend']==0]['PSAR']
    
    plt.figure(figsize=(12, 8))
    plt.plot(data['Close'], label='Close', linewidth=1)
    plt.scatter(psar_bull.index, psar_bull, color='#26a69a', label='Up Trend')
    plt.scatter(psar_bear.index, psar_bear, color='#f44336', label='Down Trend')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.title(f'{asset} Price and Parabolic SAR')
    plt.legend()
    plt.show()


def get_SAR(df_idx_high_daily , df_idx_low_daily ,df_prices_now ):
    df_SAR                          = defaultdict()
    indic                           = defaultdict()
    for asset in df_prices_now.columns:
        indic[asset]                = PSAR()
        df_SAR[asset]               = pd.DataFrame(data = 0 , index = df_idx_high_daily.index, columns = ['High','Low'])
        df_SAR[asset]['High']       = df_idx_high_daily[asset]
        df_SAR[asset]['Low']        = df_idx_low_daily[asset]
        df_SAR[asset]               = df_SAR[asset].dropna()
        
        df_SAR[asset]['PSAR']       = df_SAR[asset].apply(lambda x: indic[asset].calcPSAR(x['High'], x['Low']), axis=1)
        # Add supporting data
        df_SAR[asset]['EP']         = indic[asset].ep_list
        df_SAR[asset]['Trend']      = indic[asset].trend_list
        df_SAR[asset]['AF']         = indic[asset].af_list
        df_SAR[asset]['Close']      = df_prices_now[asset]    
    
    return df_SAR


df_SAR = get_SAR(df_idx_high_daily.copy() , df_idx_low_daily.copy() ,df_prices_now.copy() )
asset = 'USA FUTURE'
plot_SAR(df_SAR[asset].loc['2023':] , asset)










