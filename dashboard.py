import os
import numpy as np
import pandas as pd
import ccxt
import asyncio
import sqlite3

import hvplot.streamz
import hvplot.pandas
import streamz
from streamz import Stream
from streamz.dataframe import DataFrame
import panel as pn
import datetime as dt
import plotly.express as px
from joblib import dump, load
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import score
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
pn.extension()



def initialize():
    db = sqlite3.connect("etc_db.sqlite")
    with db:
        cur = db.cursor()
        cur.execute("DROP TABLE IF EXISTS etc_data")

    btc_db = sqlite3.connect("btc_db.sqlite")
    with btc_db:
        btc_cur = btc_db.cursor()
        btc_cur.execute("DROP TABLE IF EXISTS btc_data")
    


    #ETC INFO:    
    data_stream = Stream()
    data_example = pd.DataFrame(
        data={"close": []}, columns=["close"], index=pd.DatetimeIndex([])
    )
    data_stream_df = DataFrame(data_stream, example=data_example)
   
    signals_stream = Stream()
    columns = ['close','bollinger_mid_band','bollinger_upper_band','bollinger_lower_band']
    data = {'close':[],'bollinger_mid_band':[],'bollinger_upper_band':[],'bollinger_lower_band':[]}

    signals_example = pd.DataFrame(
        data=data, columns=columns, index=pd.DatetimeIndex([])
    )
    
    signals_stream_df = DataFrame(signals_stream, example=signals_example)

    #BTC INFO:
    
    btc_data_stream = Stream()
    btc_data_example = pd.DataFrame(
        data={"close": []}, columns=["close"], index=pd.DatetimeIndex([])
    )
    btc_data_stream_df = DataFrame(btc_data_stream, example=btc_data_example)
   
    btc_signals_stream = Stream()
    btc_columns = ['close','bollinger_mid_band','bollinger_upper_band','bollinger_lower_band']
    btc_data = {'close':[],'bollinger_mid_band':[],'bollinger_upper_band':[],'bollinger_lower_band':[]}

    btc_signals_example = pd.DataFrame(
        data=btc_data, columns=btc_columns, index=pd.DatetimeIndex([])
    )
    btc_signals_stream_df = DataFrame(btc_signals_stream, example=btc_signals_example)


    # Initialize Streaming DataFrames for both currencies
    dashboard = build_dashboard(data_stream_df, signals_stream_df, btc_data_stream_df, btc_signals_stream_df).servable()
    return db, btc_db, data_stream, signals_stream, btc_data_stream, btc_signals_stream, dashboard

def build_dashboard(data_stream, signals_stream, btc_data_stream, btc_signals_stream):
    """Build the dashboard."""
    color_cycle = ['#6C5B7B', '#C06C84','#355C7D','#F67280']
    #Building Static Data First
    #ETC INFO:    
    
    daily_df = pd.read_csv('etc.csv',parse_dates=True, index_col=('date'))
    daily_df.dropna(inplace=True)
    daily_df.rename(columns={'date':'Date','PriceUSD':'close'}, inplace=True)
    bollinger_window = 20
    model = load('etc_model.joblib')
    scaler = StandardScaler()

    X_scaler = scaler.fit(daily_df)
    real_future = X_scaler.transform(daily_df.tail(1))
    predictions = model.predict(real_future)
    predictions = pd.DataFrame({'Prediction':predictions,'Date':pd.datetime.today()}).set_index('Date')
    daily_df['bollinger_mid_band'] = daily_df['close'].rolling(window=bollinger_window).mean()
    daily_df['bollinger_std'] = daily_df['close'].rolling(window=20).std()

    daily_df['bollinger_upper_band']  = daily_df['bollinger_mid_band'] + (daily_df['bollinger_std'] * 1)
    daily_df['bollinger_lower_band']  = daily_df['bollinger_mid_band'] - (daily_df['bollinger_std'] * 1)

    daily_df['bollinger_long'] = np.where(daily_df['close'] < daily_df['bollinger_lower_band'], 1.0, 0.0)
    daily_df['bollinger_short'] = np.where(daily_df['close'] > daily_df['bollinger_upper_band'], -1.0, 0.0)
    daily_df['bollinger_signal'] = daily_df['bollinger_long'] + daily_df['bollinger_short']
    daily_df=daily_df[['close','bollinger_mid_band','bollinger_upper_band','bollinger_lower_band']].tail(100)
    
    
    #Building Static Data First
    #BTC INFO: 
    btc_daily_df = pd.read_csv('btc.csv',parse_dates=True, index_col=('date'))
    btc_daily_df.dropna(inplace=True)
    btc_daily_df.rename(columns={'date':'Date','PriceUSD':'close'}, inplace=True)    
    btc_daily_df['bollinger_mid_band'] = btc_daily_df['close'].rolling(window=bollinger_window).mean()
    btc_daily_df['bollinger_std'] = btc_daily_df['close'].rolling(window=20).std()

    btc_daily_df['bollinger_upper_band']  = btc_daily_df['bollinger_mid_band'] + (btc_daily_df['bollinger_std'] * 1)
    btc_daily_df['bollinger_lower_band']  = btc_daily_df['bollinger_mid_band'] - (btc_daily_df['bollinger_std'] * 1)

    btc_daily_df['bollinger_long'] = np.where(btc_daily_df['close'] < btc_daily_df['bollinger_lower_band'], 1.0, 0.0)
    btc_daily_df['bollinger_short'] = np.where(btc_daily_df['close'] > btc_daily_df['bollinger_upper_band'], -1.0, 0.0)
    btc_daily_df['bollinger_signal'] = btc_daily_df['bollinger_long'] + btc_daily_df['bollinger_short']
    btc_daily_df=btc_daily_df[['close','bollinger_mid_band','bollinger_upper_band','bollinger_lower_band']].tail(100)
    # print(predictions['Prediction'][0])


    #Building dynamic and static charts into panel
    col1= pn.Column(
        "### Yesterday's Close Price:  $" +
        
        pn.widgets.StaticText(name='Static Text', value= str(round(daily_df['close'][-1],2))).value, margin=25,
    )
    col2= pn.Column(
        "### Today's Predicted Close Price:  $" +
        pn.widgets.StaticText(name='Static Text', value=str(round(predictions['Prediction'][0],2))).value, margin=(25,50),
    )
  
    etc_row = pn.Row(
        col1,
        col2, 
    )
    col4 = pn.Column(
        data_stream.hvplot(kind='line',title="Live Price Ticker", alpha=.7, c='#355C7D',yformatter='$%.2f',ylabel='Last Price')
        .opts(width=700,bgcolor='#E4FFEE',fontsize={'title': 16, 'labels': 14}),
    )
    col5 = pn.Column(
        signals_stream.hvplot(kind='line',title='Live Price Bollinger',y=['bollinger_mid_band', 'bollinger_upper_band','bollinger_lower_band'], alpha=.2,color=color_cycle,yformatter='$%.2f')
        .opts(bgcolor='#E4FFEE',width=700, fontsize={'title': 16},legend_position='top_left')
        * signals_stream.hvplot(kind='scatter',y=['close'],color='#355C7D',yformatter='$%.2f').opts(width=700),
    )
    etc_row1 = pn.Row(
        col4,
        col5
    )
    etc_col = pn.Column(
        "# Ethereum Classic Purchase Dashboard ",
        pn.Column(pn.pane.PNG('etc.png').object), 

        etc_row,
        etc_row1,
        daily_df.hvplot(kind='line',y=['bollinger_mid_band', 'bollinger_upper_band','bollinger_lower_band'],title="Standard Bollinger Bands", alpha=.7,color=color_cycle,yformatter='$%.2f',ylabel='Last Price')
        .opts(width=1400,bgcolor='#E4FFEE',fontsize={'title': 16, 'labels': 14},legend_position='top_left')
        * daily_df.hvplot(kind='line',y='close',title="Standard Bollinger Bands", alpha=.7,color=color_cycle,yformatter='$%.2f',ylabel='Last Price')
        .opts(line_dash='dashed',width=1400,bgcolor='#E4FFEE',fontsize={'title': 16, 'labels': 14}),


    )
    #BTC INFO
    btc_col1= pn.Column(
        "### Yesterday's Close Price:  $" +
        
        pn.widgets.StaticText(name='Static Text', value= str(round(btc_daily_df['close'][-1],2))).value, margin=25,
    )
   
  
    btc_row = pn.Row(
        btc_col1,
         
    )
    btc_col4 = pn.Column(
        btc_data_stream.hvplot(kind='line',title="Live Price Ticker", alpha=.7, c='#355C7D',yformatter='$%.2f',ylabel='Last Price')
        .opts(width=700,bgcolor='#FEF0E0',fontsize={'title': 16, 'labels': 14}),
    )
    btc_col5 = pn.Column(
        btc_signals_stream.hvplot(kind='line',title='Live Price Bollinger',y=['bollinger_mid_band', 'bollinger_upper_band','bollinger_lower_band'], alpha=.2,color=color_cycle,yformatter='$%.2f')
        .opts(bgcolor='#FEF0E0',width=700, fontsize={'title': 16},legend_position='top_left')
        * btc_signals_stream.hvplot(kind='scatter',y=['close'],color='#355C7D',yformatter='$%.2f').opts(width=700),
    )
    btc_row1 = pn.Row(
        btc_col4,
        btc_col5
    )
    btc_col = pn.Column(
        "# Bitcoin Purchase Dashboard ",
        pn.Column(pn.pane.PNG('btc.png').object), 

        btc_row,
        btc_row1,
        btc_daily_df.hvplot(kind='line',y=['bollinger_mid_band', 'bollinger_upper_band','bollinger_lower_band'],title="Standard Bollinger Bands", alpha=.7,color=color_cycle,yformatter='$%.2f',ylabel='Last Price')
        .opts(width=1400,bgcolor='#FEF0E0',fontsize={'title': 16, 'labels': 14},legend_position='top_left')
        * btc_daily_df.hvplot(kind='line',y='close',title="Standard Bollinger Bands", alpha=.7,color=color_cycle,yformatter='$%.2f',ylabel='Last Price')
        .opts(line_dash='dashed',width=1400,bgcolor='#FEF0E0',fontsize={'title': 16, 'labels': 14}),


    )
  
#     btc_col = pn.Column(
#         "# Bitcoin Purchase Dashboard",
#         "## Yesterday's Close Price:  $" +
#         pn.widgets.StaticText(name='Static Text', value=str(btc_daily_df['close'][0])).value,        
#         btc_data_stream.hvplot(kind='line',title="Current Price", alpha=.7, color=color_cycle,yformatter='$%.2f',ylabel='Last Price')
#             .opts(width=1400,bgcolor='#FDE6DD',fontsize={'title': 16, 'labels': 14}),

#         btc_signals_stream.hvplot(kind='line',title='Live Price Bollinger',y=['bollinger_mid_band', 'bollinger_upper_band','bollinger_lower_band'], alpha=.7,color=color_cycle,yformatter='$%.2f')
#         .opts(bgcolor='#FDE6DD',width=1400, fontsize={'title': 16})
#         * btc_signals_stream.hvplot(kind='scatter',y=['close'],color='#355C7D',yformatter='$%.2f').opts(width=1400),
#         btc_daily_df.hvplot(kind='line',y=['bollinger_mid_band', 'bollinger_upper_band','bollinger_lower_band'],title="Standard Bollinger Bands", alpha=.7,color=color_cycle,yformatter='$%.2f',ylabel='Last Price')
#         .opts(width=1400,bgcolor='#FDE6DD',fontsize={'title': 16, 'labels': 14})
#         * btc_daily_df.hvplot(kind='line',y='close',title="Standard Bollinger Bands", alpha=.7,color=color_cycle,yformatter='$%.2f',ylabel='Last Price')
#         .opts(line_dash='dashed',width=1400,bgcolor='#FDE6DD',fontsize={'title': 16, 'labels': 14})
# ,
#     )

    
    dashboard = pn.Tabs(
    ("Ethereum Classic",etc_col),
    ("Bitcoin", btc_col)
    )


    return dashboard
def fetch_data():
    #Set kraken API variables:
    kraken_public_key = os.getenv("KRAKEN_PUBLIC_KEY")
    kraken_secret_key = os.getenv("KRAKEN_SECRET_KEY")
    kraken = ccxt.kraken({"apiKey": kraken_public_key, "secret": kraken_secret_key})
   
    #ETC INFO:
    close = kraken.fetch_ticker("ETC/USD")["last"]
    datetime = kraken.fetch_ticker("ETC/USD")["datetime"]
    open1 = kraken.fetch_ticker("ETC/USD")["open"]
    last = kraken.fetch_ticker("ETC/USD")["last"]
    df = pd.DataFrame({"close": [last]})
    df.index = pd.to_datetime([datetime])

    return df
     
def btc_fetch_data():

    #Set kraken API variables:
    kraken_public_key = os.getenv("KRAKEN_PUBLIC_KEY")
    kraken_secret_key = os.getenv("KRAKEN_SECRET_KEY")
    kraken = ccxt.kraken({"apiKey": kraken_public_key, "secret": kraken_secret_key})
    
    #BTC INFO:
    btc_close = kraken.fetch_ticker("BTC/USD")["last"]
    btc_datetime = kraken.fetch_ticker("BTC/USD")["datetime"]
    btc_open1 = kraken.fetch_ticker("BTC/USD")["open"]
    btc_last = kraken.fetch_ticker("BTC/USD")["last"]
    btc_df = pd.DataFrame({"close": [btc_last]})
    btc_df.index = pd.to_datetime([btc_datetime])
    return btc_df

def generate_signals(df):
    bollinger_window = 60
    signals = df
    
    #ETC INFO:    
    signals['bollinger_mid_band'] = signals['close'].rolling(window=bollinger_window).mean()
    signals['bollinger_std'] = signals['close'].rolling(window=60).std()

    signals['bollinger_upper_band']  = signals['bollinger_mid_band'] + (signals['bollinger_std'] * 1)
    signals['bollinger_lower_band']  = signals['bollinger_mid_band'] - (signals['bollinger_std'] * 1)

    signals['bollinger_long'] = np.where(signals['close'] < signals['bollinger_lower_band'], 1.0, 0.0)
    signals['bollinger_short'] = np.where(signals['close'] > signals['bollinger_upper_band'], -1.0, 0.0)
    signals['bollinger_signal'] = signals['bollinger_long'] + signals['bollinger_short']
    signals=signals[['close','bollinger_mid_band','bollinger_upper_band','bollinger_lower_band']]
    return signals

def btc_generate_signals(btc_df):
    #BTC INFO:
    bollinger_window = 60
    btc_signals = btc_df

    btc_signals['bollinger_mid_band'] = btc_signals['close'].rolling(window=bollinger_window).mean()
    btc_signals['bollinger_std'] = btc_signals['close'].rolling(window=60).std()

    btc_signals['bollinger_upper_band']  = btc_signals['bollinger_mid_band'] + (btc_signals['bollinger_std'] * 1)
    btc_signals['bollinger_lower_band']  = btc_signals['bollinger_mid_band'] - (btc_signals['bollinger_std'] * 1)

    btc_signals['bollinger_long'] = np.where(btc_signals['close'] < btc_signals['bollinger_lower_band'], 1.0, 0.0)
    btc_signals['bollinger_short'] = np.where(btc_signals['close'] > btc_signals['bollinger_upper_band'], -1.0, 0.0)
    btc_signals['bollinger_signal'] = btc_signals['bollinger_long'] + btc_signals['bollinger_short']
    btc_signals=btc_signals[['close','bollinger_mid_band','bollinger_upper_band','bollinger_lower_band']]

    return btc_signals

db, btc_db, data_stream, signals_stream, btc_data_stream, btc_signals_stream, dashboard=initialize()
# print(data_stream)
# print(signals)
# dashboard.servable()
async def main():
    loop = asyncio.get_event_loop()

    while True:
       
        global data_stream
        global signals_stream
        global btc_data_stream
        global btc_signals_stream
       
        
        #ETC INFO        
        new_df = await loop.run_in_executor(None, fetch_data)
        new_df.to_sql("etc_data", db, if_exists="append", index=True)
        df = pd.read_sql(f"select * from etc_data limit 1000", db)
        signals = generate_signals(df)
        signals_stream.emit(signals)
        data_stream.emit(new_df)


        #BTC INFO
        btc_new_df = await loop.run_in_executor(None, btc_fetch_data)
        btc_new_df.to_sql("btc_data", btc_db, if_exists="append", index=True)
        btc_df = pd.read_sql(f"select * from btc_data limit 1000", btc_db)
        btc_signals = btc_generate_signals(btc_df)
        btc_signals_stream.emit(btc_signals)
        btc_data_stream.emit(btc_new_df)


        print('test')
        # Update the Dashboard
        await asyncio.sleep(1)


# Python 3.7+
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
