import yfinance as yf
import pandas as pd
import mplfinance as mpf
import os
from random import sample
import glob
import matplotlib.pyplot as plt
import csv

pd.options.mode.chained_assignment = None # Deaktivierung der Warnungen
############################## Funktion zur Datenbeschaffung und Speicherung ##############################

def get_stock_data(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date, interval='1d')
    return data

def save_chart(data, symbol, output_dir, chart_type, start_date):
    fig, ax = plt.subplots(figsize=(10, 6))
    mpf.plot(data[-100:], type='candle', style='classic', ax=ax, volume=False, show_nontrading=False)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_ylabel('')
    plt.savefig(os.path.join(output_dir, f'CHART_{symbol}_{chart_type}_{start_date}.png'), bbox_inches='tight')
    plt.close()

def detect_hammer(data):
    body = data['Close'] - data['Open']
    lower_shadow = data['Open'] - data['Low']
    upper_shadow = data['High'] - data['Close']
    current_open = data['Open']
    current_close = data['Close']
    prev3Close = data['Close'].shift(3)
    prev3Open = data['Open'].shift(3)
    prev2Close = data['Close'].shift(2)
    prev2Open = data['Open'].shift(2)
    prevClose = data['Close'].shift(1)
    prevOpen = data['Open'].shift(1)
    futureOpen = data['Open'].shift(-1)
    futureClose = data['Close'].shift(-1)

    condition1 = (lower_shadow > 2 * body)
    condition2 = (prev2Close < prev2Open)
    condition3 = (prevClose < prevOpen)
    condition4 = (current_open < current_close)
    condition5 = (prev3Close < prev3Open)
    condition6 = (futureOpen < futureClose)
    condition7 = (upper_shadow < 0.5 * body)

    sum_condition = condition1 & condition2 & condition3 & condition4 & condition5 & condition6 & condition7
    if sum_condition.any():
        result = data.loc[sum_condition]
        index = result.index[0]
        index_datetime = pd.to_datetime(index)  # Umwandlung in ein Timestamp-Objekt
        result_and_surrounding = data.loc[index_datetime - pd.Timedelta(days=15) : index_datetime + pd.Timedelta(days=15)]
        return result_and_surrounding
    else:
        return False
   
def detect_shootingstar(data):
    body = data['Open'] - data['Close']
    lower_shadow = data['Close'] - data['Low']
    upper_shadow = data['High'] - data['Open']
    current_open = data['Open']
    current_close = data['Close']
    prev3Close = data['Close'].shift(3)
    prev3Open = data['Open'].shift(3)
    prev2Close = data['Close'].shift(2)
    prev2Open = data['Open'].shift(2)
    prevClose = data['Close'].shift(1)
    prevOpen = data['Open'].shift(1)
    futureOpen = data['Open'].shift(-1)
    futureClose = data['Close'].shift(-1)

    condition1 = (prev3Close > prev3Open)
    condition2 = (prev2Close > prev2Open)
    condition3 = (prevClose > prevOpen)
    condition4 = (futureClose < futureOpen)
    condition5 = (current_close < current_open)
    condition6 = (upper_shadow > 2*body)
    condition7 = (lower_shadow < 0.5*body)
    
    sum_condition = condition1 & condition2 & condition3 & condition4 & condition5 & condition6 & condition7
    if sum_condition.any():
        result = data.loc[sum_condition]
        index = result.index[0]
        index_datetime = pd.to_datetime(index)  # Umwandlung in ein Timestamp-Objekt
        result_and_surrounding = data.loc[index_datetime - pd.Timedelta(days=15) : index_datetime + pd.Timedelta(days=15)]
        return result_and_surrounding
    else:
        return False

def detect_blackcrow(data):
    prev5Close = data['Close'].shift(5)
    prev5Open = data['Open'].shift(5)
    prev4Close = data['Close'].shift(4)
    prev4Open = data['Open'].shift(4)
    prev3Close = data['Close'].shift(3)
    prev3Open = data['Open'].shift(3)
    prev2Close = data['Close'].shift(2)
    prev2Open = data['Open'].shift(2)
    prevClose = data['Close'].shift(1)
    prevOpen = data['Open'].shift(1)
    current_open = data['Open']
    current_close = data['Close']
    futureOpen = data['Open'].shift(-1)
    futureClose = data['Close'].shift(-1)
    futureClose2 = data['Close'].shift(-2)
    futureOpen2 = data['Open'].shift(-2)
    
    condition1 = (prev5Close > prev5Open)
    condition2 = (prev4Close > prev4Open)
    condition3 = (prev3Close > prev3Open)
    condition4 = (prev2Close < prev2Open)
    condition5 = (prevClose < prevOpen)
    condition6 = (current_close < current_open)
    condition7 = (futureClose < futureOpen)
    condition8 = (current_open < prevOpen) & (current_open > prevClose)
    condition9 = (futureOpen < current_open) & (futureOpen > current_close)
    condition10 = (futureClose2 < futureOpen2)

    sum_condition = condition1 & condition2 & condition3 & condition4 & condition5 & condition6 & condition7 & condition8 & condition9 & condition10
    if sum_condition.any():
        result = data.loc[sum_condition]
        index = result.index[0]
        index_datetime = pd.to_datetime(index)  # Umwandlung in ein Timestamp-Objekt
        result_and_surrounding = data.loc[index_datetime - pd.Timedelta(days=15) : index_datetime + pd.Timedelta(days=15)]
        return result_and_surrounding
    else:
        return False

def detect_whitesoldier(data):
    prev5Close = data['Close'].shift(5)
    prev5Open = data['Open'].shift(5)
    prev4Close = data['Close'].shift(4)
    prev4Open = data['Open'].shift(4)
    prev3Close = data['Close'].shift(3)
    prev3Open = data['Open'].shift(3)
    prev2Close = data['Close'].shift(2)
    prev2Open = data['Open'].shift(2)
    prevClose = data['Close'].shift(1)
    prevOpen = data['Open'].shift(1)
    current_open = data['Open']
    current_close = data['Close']
    futureOpen = data['Open'].shift(-1)
    futureClose = data['Close'].shift(-1)
    futureClose2 = data['Close'].shift(-2)
    futureOpen2 = data['Open'].shift(-2)

    condition1 = (prev5Close < prev5Open)
    condition2 = (prev4Close < prev4Open)
    condition3 = (prev3Close < prev3Open)
    condition4 = (prev2Close > prev2Open)
    condition5 = (prevClose > prevOpen)
    condition6 = (current_close > current_open)
    condition7 = (futureClose > futureOpen)
    condition8 = (current_open > prevOpen) & (current_open < prevClose)
    condition9 = (futureOpen > current_open) & (futureOpen < current_close)
    condition10 = (futureClose2 > futureOpen2)

    sum_condition = condition1 & condition2 & condition3 & condition4 & condition5 & condition6 & condition7 & condition8 & condition9 & condition10
    if sum_condition.any():
        result = data.loc[sum_condition]
        index = result.index[0]
        index_datetime = pd.to_datetime(index)  # Umwandlung in ein Timestamp-Objekt
        result_and_surrounding = data.loc[index_datetime - pd.Timedelta(days=15) : index_datetime + pd.Timedelta(days=15)]
        return result_and_surrounding
    else:
        return False

def detect_bengulfing(data):
    prev3Close = data['Close'].shift(3)
    prev3Open = data['Open'].shift(3)
    prev2Close = data['Close'].shift(2)
    prev2Open = data['Open'].shift(2)
    prevClose = data['Close'].shift(1)
    prevOpen = data['Open'].shift(1)
    current_open = data['Open']
    current_close = data['Close']
    current_HIGH = data['High']
    current_LOW = data['Low']
    futureOpen = data['Open'].shift(-1)
    futureClose = data['Close'].shift(-1)
    futureClose2 = data['Close'].shift(-2)
    futureOpen2 = data['Open'].shift(-2)
    futureClose3 = data['Close'].shift(-3)
    futureOpen3 = data['Open'].shift(-3)

    condition7 = (prev3Close < prev3Open)
    condition8 = (prev2Close < prev2Open)
    condition1 = (current_close < current_open)
    condition2 = (prevClose < prevOpen)
    condition3 = (futureOpen < futureClose)
    condition4 = (futureOpen2 < futureClose2)
    condition9 = (futureOpen3 < futureClose3)
    condition5 = futureOpen.between( current_close - 0.01*current_close, current_close + 0.01*current_close)
    condition6 = futureClose > current_open + 0.01*current_open
    sum_condition = condition1 & condition2 & condition3 & condition4 & condition5 &  condition6 & condition7 & condition8 & condition9

    if sum_condition.any():
        result = data.loc[sum_condition]
        index = result.index[0]
        index_datetime = pd.to_datetime(index)  # Umwandlung in ein Timestamp-Objekt
        result_and_surrounding = data.loc[index_datetime - pd.Timedelta(days=15):index_datetime + pd.Timedelta(days=15)]
        return result_and_surrounding
    else:
        return None

def detect_bhamari(data):
    prev5Close = data['Close'].shift(5)
    prev5Open = data['Open'].shift(5)
    prev4Close = data['Close'].shift(4)
    prev4Open = data['Open'].shift(4)
    prev3Close = data['Close'].shift(3)
    prev3Open = data['Open'].shift(3)
    prev2High = data['High'].shift(2)
    prev2Low = data['Low'].shift(2)
    prevClose = data['Close'].shift(1)
    prevOpen = data['Open'].shift(1)
    current_open = data['Open']
    current_close = data['Close']
    futureOpen = data['Open'].shift(-1)
    futureClose = data['Close'].shift(-1)


    condition1 = (prev5Close < prev5Open)
    condition2 = (prev4Close < prev4Open)
    condition3 = (prev3Close < prev3Open)
    condition4 = prev2Low.between(prev3Close, prev3Open)
    condition5 = prev2High.between(prev3Close, prev3Open)
    condition6 = (prevClose > prevOpen)
    condition7 = (current_close > current_open)
    condition8 = (futureClose > futureOpen)

    sum_condition = condition1 & condition2 & condition3 & condition4 & condition5 & condition6 & condition7 & condition8
    if sum_condition.any():
        result = data.loc[sum_condition]
        index = result.index[0]
        index_datetime = pd.to_datetime(index)  # Umwandlung in ein Timestamp-Objekt
        result_and_surrounding = data.loc[index_datetime - pd.Timedelta(days=15) : index_datetime + pd.Timedelta(days=15)]
        return result_and_surrounding
    else:
        return False    

def detect_morningstar(data):
    prev5Close = data['Close'].shift(5)
    prev5Open = data['Open'].shift(5)
    prev4Close = data['Close'].shift(4)
    prev4Open = data['Open'].shift(4)
    prev3Close = data['Close'].shift(3)
    prev3Open = data['Open'].shift(3)
    prev3Close = data['Close'].shift(3)
    prev3Open = data['Open'].shift(3)
    prev2Close = data['Close'].shift(2)
    prev2Open = data['Open'].shift(2)
    prev2_high = data['High'].shift(2)
    prevClose = data['Close'].shift(1)
    prevOpen = data['Open'].shift(1)
    current_open = data['Open']
    current_close = data['Close']
    current_high = data['High']
    futureOpen = data['Open'].shift(-1)
    futureClose = data['Close'].shift(-1)
    futureClose2 = data['Close'].shift(-2)
    futureOpen2 = data['Open'].shift(-2)
    # Genereller Verlauf der Kerzen
    condition1 = (prev3Close < prev3Open)
    condition2 = (prev2Close < prev2Open)
    condition3 = (prevClose > prevOpen)
    condition4 = (current_close > current_open)
    condition5 = (futureClose > futureOpen)
    condition16 = (futureClose2 > futureOpen2)
    condition14 = (prev5Close < prev5Open)
    condition15 = (prev4Close < prev4Open)
    # Close der Bullishen Kerze in der Nähe des Opens und Highs der bearishen Kerze
    condition6 = current_close.between(prev2Open - 0.01 * prev2Open, prev2Open + 0.01 * prev2Open)
    condition7 = current_close.between(prev2_high - 0.02 * prev2_high, prev2_high + 0.02 * prev2_high)
    condition12 = current_high.between(prev2Open - 0.02 * prev2Open, prev2Open + 0.02 * prev2Open)
    condition13 = current_high.between(prev2_high - 0.03 * prev2_high, prev2_high + 0.03 * prev2_high)
    
    Band  = 0.005
    condition8 = prev2Close.between(prevOpen - prevOpen*Band, prevOpen + prevOpen*Band)
    condition9 = current_open.between(prevOpen - prevOpen*Band, prevOpen + prevOpen*Band)
    condition10 = prev2Close.between(prevClose - prevClose*Band, prevClose + prevClose*Band)
    condition11 = current_open.between(prevClose - prevClose*Band, prevClose + prevClose*Band)

    sum_condition = condition1 & condition2 & condition3 & condition4 & condition5 & condition6 & condition7 & condition8 & condition9 & condition10 & condition11 & condition12 & condition13 & condition14 & condition15 & condition16
    if sum_condition.any():
        result = data.loc[sum_condition]
        index = result.index[0]
        index_datetime = pd.to_datetime(index)  # Umwandlung in ein Timestamp-Objekt
        result_and_surrounding = data.loc[index_datetime - pd.Timedelta(days=15) : index_datetime + pd.Timedelta(days=15)]
        return result_and_surrounding
    else:
        return False

OD_Hammer_Pattern = 'Hammer_Pattern'
if not os.path.exists(OD_Hammer_Pattern):
    os.makedirs(OD_Hammer_Pattern)

OD_ShootingStar_Pattern = 'ShootingStar_Pattern'
if not os.path.exists(OD_ShootingStar_Pattern):
    os.makedirs(OD_ShootingStar_Pattern)

OD_BlackCrows_Pattern = 'BlackCrows_Pattern'
if not os.path.exists(OD_BlackCrows_Pattern):
    os.makedirs(OD_BlackCrows_Pattern)

OD_WhiteSoldier_Pattern = 'WhiteSoldier_Pattern'
if not os.path.exists(OD_WhiteSoldier_Pattern):
    os.makedirs(OD_WhiteSoldier_Pattern)

OD_BENGULFING_Pattern = 'Engulfing_Pattern'
if not os.path.exists(OD_BENGULFING_Pattern):
    os.makedirs(OD_BENGULFING_Pattern)

OD_BHAMARI_Pattern = 'Hamari_Pattern'
if not os.path.exists(OD_BHAMARI_Pattern):
    os.makedirs(OD_BHAMARI_Pattern)

OD_MORNINGSTAR_Pattern = 'Morningstar_Pattern'
if not os.path.exists(OD_MORNINGSTAR_Pattern):
    os.makedirs(OD_MORNINGSTAR_Pattern)

# Aktien Ticker Liste von ChatGPT erstellen lassen
Ticker = [
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'FB', 'V', 'JPM', 'JNJ', 'PYPL', 'NVDA', 'PG', 'MA', 'INTC', 'CSCO', 'ASML', 'CMCSA', 'PFE', 'T', 'NFLX',
    'ABBV', 'WMT', 'BAC', 'ADBE', 'XOM', 'DIS', 'ABT', 'C', 'VZ', 'CVX', 'KO', 'CRM', 'IBM', 'GS', 'MDT', 'ORCL', 'DHR', 'INTU', 'HON', 'NEE', 'BA', 'AVGO',
    'AMGN', 'ACN', 'MMM', 'TXN', 'QCOM', 'COST', 'GE', 'WFC', 'UNH', 'NKE', 'LMT', 'TMO', 'AXP', 'MO', 'LRCX', 'SBUX', 'LOW', 'PM', 'UNP', 'UPS', 'CAT',
    'AMAT', 'CVS', 'IBM', 'VLO', 'USB', 'WM', 'BLK', 'MET', 'DUK', 'TJX', 'GS', 'MCD', 'TGT', 'FDX', 'GS', 'EMR', 'CME', 'BKNG', 'ANTM', 'AAP', 'ADI', 'AIG',
    'ALL', 'AMT', 'APD', 'AON', 'AXP', 'BA', 'BAC', 'BAX', 'BDX', 'BIIB', 'BK', 'BLK', 'BMY', 'BRK.B', 'GOOG', 'AAP', 'AES', 'AFL', 'A', 'APD', 'AKAM', 'ALK',
    'ALB', 'ARE', 'ALXN', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'ABC', 'AME',
    'AMGN', 'APH', 'ADI', 'ANSS', 'ANTM', 'AON', 'APA', 'AAPL', 'AMAT', 'APTV', 'ADM', 'ANET', 'AJG', 'AIZ', 'ATO', 'T', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY',
    'BKR', 'BLL', 'BAC', 'BK', 'BAX', 'BDX', 'BRK.B', 'BBY', 'BIO', 'TECH', 'BIIB', 'BLK', 'BA', 'BKNG', 'BWA', 'BXP', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'BF.B',
    'CHRW', 'COG', 'CDNS', 'CZR', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CTLT', 'CAT', 'CBOE', 'CBRE', 'CDW', 'CE', 'CNC', 'CNP', 'CERN', 'CF', 'SCHW',
    'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CTXS', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG', 'COP',
    'ED', 'STZ', 'COO', 'CPRT', 'GLW', 'CTVA', 'COST', 'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DAL', 'XRAY', 'DVN', 'DXCM', 'FANG', 'DLR',
    'DFS', 'DISCA', 'DISCK', 'DISH', 'DG', 'DLTR', 'D', 'DPZ', 'DOV', 'DOW', 'DTE', 'DUK', 'DRE', 'DD', 'DXC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMR',
    'ENPH', 'ETR', 'EOG', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ETSY', 'EVRG', 'ES', 'RE', 'EXC', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FB', 'FAST', 'FRT', 'FDX',
    'FIS', 'FITB', 'FE', 'FRC', 'FISV', 'FLT', 'FLIR', 'FLS', 'FMC', 'F', 'FTNT', 'FTV', 'FBHS', 'FOXA', 'FOX', 'BEN', 'FCX', 'GPS', 'GRMN', 'IT', 'GD', 'GE', 'GIS',
    'GM', 'GPC', 'GILD', 'GL', 'GPN', 'GS', 'GWW', 'HAL', 'HBI', 'HIG', 'HAS', 'HCA', 'PEAK', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HFC', 'HOLX', 'HD', 'HON', 'HRL',
    'HST', 'HWM', 'HPQ', 'HUM', 'HBAN', 'HII', 'IEX', 'IDXX', 'INFO', 'ITW', 'ILMN', 'INCY', 'IR', 'INTC', 'ICE', 'IBM', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ',
    'IPGP', 'IQV', 'IRM', 'JKHY', 'J', 'JBHT', 'SJM', 'JNJ', 'JCI', 'JPM', 'JNPR', 'KSU', 'K', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KLAC', 'KHC', 'KR', 'LB', 'LHX',
    'LH', 'LRCX', 'LW', 'LVS', 'LEG', 'LDOS', 'LEN', 'LLY', 'AAP', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALK', 'ALL', 'ALLE', 'ALXN', 'AMAT', 'AMCR', 'AME', 'AMED',
    'AMG', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APTV', 'ARE', 'ATO', 'ATVI', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXP', 'AZO',
    'BA', 'BAC', 'BAX', 'BBY', 'BDX', 'BEN', 'BF.B', 'BIIB', 'BIO', 'BIOTECH', 'BK', 'BKNG', 'BKR', 'BLK', 'BLL', 'BMY', 'BR', 'BRK.B', 'BRO', 'BSX', 'BWA', 'BXP',
    'C', 'CAG', 'CAH', 'CARR', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CE', 'CERN', 'CF', 'CFG', 'CHD', 'CHRW', 'CI', 'CINF', 'CL', 'CLX', 'CMA',
    'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COG', 'COO', 'COP', 'COST', 'CPB', 'CPRT', 'CRM', 'CRWD', 'CSX', 'CTAS', 'CTSH', 'CTVA', 'CTXS', 'CVS',
    'CVX', 'CZR', 'D', 'DAL', 'DD', 'DE', 'DFS', 'DG', 'DHI', 'DHR', 'DIS', 'DISCA', 'DISCK', 'DISH', 'DLR', 'DLTR', 'D', 'DPZ', 'DOV', 'DOW', 'DTE', 'DUK', 'DRE',
    'DD', 'DXC', 'EMN', 'ETN', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'EMN', 'EMR', 'ENPH', 'ETR', 'EOG', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ETSY', 'EVRG', 'ES',
    'RE', 'EXC', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FB', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FE', 'FRC', 'FISV', 'FLT', 'FLIR', 'FLS', 'FMC', 'F', 'FTNT', 'FTV',
    'FBHS', 'FOXA', 'FOX', 'BEN', 'FCX', 'GPS', 'GRMN', 'IT', 'GD', 'GE', 'GIS', 'GM', 'GPC', 'GILD', 'GL', 'GPN', 'GS', 'GWW', 'HAL', 'HBI', 'HIG', 'HAS', 'HCA',
    'PEAK', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HFC', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUM', 'HBAN', 'HII', 'IEX', 'IDXX', 'INFO', 'ITW', 'ILMN',
    'INCY', 'IR', 'INTC', 'ICE', 'IBM', 'IP', 'IPG', 'IFF', 'INTU', 'IPGP', 'IQV', 'IRM', 'JKHY', 'J', 'JBHT', 'SJM', 'JNJ', 'JCI', 'JPM', 'JNPR', 'KSU', 'K', 'KEY',
    'KEYS', 'KMB', 'KIM', 'KLAC', 'KHC', 'KR', 'LB', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LEG', 'LDOS', 'LEN', 'LLY', 'AAP', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALK',
    'ALL', 'ALLE', 'ALXN', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'ABC', 'AME', 'AMGN',
    'APH', 'ADI', 'ANSS', 'ANTM', 'AON', 'APA', 'AAPL', 'AMAT', 'APTV', 'ADM', 'ANET', 'AJG', 'AIZ', 'ATO', 'T', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'BKR', 'BLL', 'BAC',
    'BK', 'BAX', 'BDX', 'BRK.B', 'BBY', 'BIO', 'TECH', 'BIIB', 'BLK', 'BA', 'BKNG', 'BWA', 'BXP', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'BF.B', 'CHRW', 'COG', 'CDNS',
    'CZR', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CTLT', 'CAT', 'CBOE', 'CBRE', 'CDW', 'CE', 'CNC', 'CNP', 'CERN', 'CF', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB',
    'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CTXS', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG', 'COP', 'ED', 'STZ', 'COO', 'CPRT', 'GLW',
    'CTVA', 'COST', 'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DAL', 'XRAY', 'DVN', 'DXCM', 'FANG', 'DLR', 'DFS', 'DISCA', 'DISCK', 'DISH', 'DG',
    'DLTR', 'D', 'DPZ', 'DOV', 'DOW', 'DTE', 'DUK', 'DRE', 'DD', 'DXC', 'EM']

for index, symbol in enumerate(Ticker):
    ######### kurzere Zeiträume #########
    start_date = '2023-11-01'
    end_date = '2023-12-28'

    data = get_stock_data(symbol, start_date, end_date)
    if len(data) > 15:
        ##### ----------- Hammer Pattern ----------- #####
        Hammer_data = detect_hammer(data)
        if isinstance(Hammer_data, pd.DataFrame):
            save_chart(Hammer_data, symbol, OD_Hammer_Pattern, 'Hammer', start_date)
            print(f"{index} -> Hammer Pattern Detected")

        ##### ----------- ShootingStar Pattern ----------- #####   
        SS_data = detect_shootingstar(data)
        if isinstance(SS_data, pd.DataFrame):
            save_chart(SS_data, symbol, OD_ShootingStar_Pattern, 'ShootingStar', start_date)
            print(f"{index} -> Shooting Star Pattern Detected")

        ##### ----------- BlackCrow Pattern ----------- #####   
        BC_data = detect_blackcrow(data)
        if isinstance(BC_data, pd.DataFrame):
            save_chart(BC_data, symbol, OD_BlackCrows_Pattern, 'BlackCrow', start_date)
            print(f"{index} -> Black Crow Pattern Detected")

        ##### ----------- WhiteSoldier Pattern ----------- #####   
        WS_data = detect_whitesoldier(data)
        if isinstance(WS_data, pd.DataFrame):
            save_chart(WS_data, symbol, OD_WhiteSoldier_Pattern, 'WhiteSoldier', start_date)
            print(f"{index} -> White Soldier Pattern Detected")

        ##### ----------- Bullish Engulfing Pattern ----------- #####   
        BE_data = detect_bengulfing(data)
        if isinstance(BE_data, pd.DataFrame):
            save_chart(BE_data, symbol, OD_BENGULFING_Pattern, 'BEngulfing', start_date)
            print(f"{index} -> Engulfing Pattern Detected")
        
        ##### ----------- Bullish Hamari Pattern ----------- #####   
        BH_data = detect_bhamari(data)
        if isinstance(BH_data, pd.DataFrame):
            save_chart(BH_data, symbol, OD_BHAMARI_Pattern, 'BHamari', start_date)
            print(f"{index} -> Hamari Pattern Detected")

        ##### ----------- MorningStar Pattern ----------- #####   
        MS_data = detect_morningstar(data)
        if isinstance(MS_data, pd.DataFrame):
            save_chart(MS_data, symbol, OD_MORNINGSTAR_Pattern, 'MorningStar', start_date)
            print(f"{index} -> MorningStar Pattern Detected")

print('Skript abgeschlossen.')