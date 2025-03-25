import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

def summarize_data(clean_data_path):
    
    df = pd.read_csv(clean_data_path, index_col = 0)
    
    df.index = pd.to_datetime(df.index)

    #print(df.head(10))

    print("\nSummary:")
    print(df.describe())

    #Correlation Matrix
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    print("\nCorrelation Matrix:")
    correlation = numeric_df.corr()
    print(correlation)
    
    return df

#Line plot of stock prices
def plot_stock_price(df):  
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['Close'], label = 'Closing Price', color = 'blue')
    plt.title('Stock Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.xticks(rotation = 45)
    plt.legend()
    plt.show()

#Candlestick Chart 
def plot_candlestick_chart(df):
    fig = go.Figure(data=[go.Candlestick(x= df.index,
                                         open=df['Open'],
                                         high = df['High'],
                                         close = df['Close'],
                                         name = 'Candlestick Chart')])
    
    fig.update_layout(title = 'Stock Price Candlestick Chart',
                      xaxis_title = 'Date',
                      yaxis_title = 'Price',
                      xaxis_rangeslider_visible=False)
    fig.show()
    