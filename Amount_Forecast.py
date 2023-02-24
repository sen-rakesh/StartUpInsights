import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import re 
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title = "Bitathon 2023")
st.markdown(""" ## Time Series Forecasting of Invested Amount using SARIMAX

Let's Go !! """)
raw_data = st.file_uploader("Upload a Dataset")
if raw_data is not None:
    data = pd.read_csv(raw_data,encoding='utf-8')
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    temp_df= data.groupby('Date')['Amount in USD'].sum().reset_index()

    temp_df = temp_df.set_index(['Date'])

    temp_df= temp_df['Amount in USD'].resample('MS').mean()
    temp_df = temp_df.fillna(temp_df.bfill())

    temp_df.plot(figsize=(10,10))
    plt.show()
 
    st.subheader('Seasonal Decomposition')
    from pylab import rcParams
    import statsmodels.api as sm
    decomposition = sm.tsa.seasonal_decompose(temp_df, model='multiplicative')

    fig = decomposition.plot()
    fig.set_size_inches((16, 9))
    plt.show()
    st.pyplot()
    
    from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
    plot_acf(temp_df,lags=20)
    plot_pacf(temp_df,lags=20)

    from statsmodels.tsa.stattools import adfuller
    rollingmean=temp_df.rolling(window=12).mean()
    rollingstd=temp_df.rolling(window=12).std()
    orig=temp_df.plot()
    mean=rollingmean.plot(label='Mean')
    std=rollingstd.plot()
    plt.legend()
    plt.show()

    def adfuller_test(demand):
        result=adfuller(demand,autolag='AIC')
        labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
        for value,label in zip(result,labels):
            print(label+' : '+str(value) )
        if result[1] <= 0.05:
            print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
        else:
            print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
    adfuller_test(temp_df)
    temp_df_logscale=np.log(temp_df)
    movingaverages=temp_df_logscale.rolling(window=12).mean()
    movingstd=temp_df_logscale.rolling(window=12).std()
    temp_df_logscale.plot()
    movingaverages.plot()
    plt.show()
    temp_dflogscaleminus=temp_df_logscale-movingaverages.shift(2)
    temp_dflogscaleminus.dropna(inplace=True)
    adfuller_test(temp_dflogscaleminus)
    import itertools
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(temp_df,
                                                order=param,
                                                seasonal_order=param_seasonal,enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                
            except:
                continue

    model=sm.tsa.statespace.SARIMAX(temp_df,order=(1,1,1),seasonal_order=(1,1,1,12))
    results=model.fit()

    st.subheader('Diagnostic plot')
    results.plot_diagnostics(figsize=(20, 10))
    plt.show()
    st.pyplot()

    pred = results.get_prediction(start=pd.to_datetime('2019-05-01'), dynamic=True) #false is when using the entire history.
    #Confidence interval.
    pred_ci = pred.conf_int()

  
  

    y_forecasted = pred.predicted_mean
    y_truth = temp_df['2020-01-01':]
    mse = ((y_forecasted - y_truth) ** 2).mean()
    st.subheader('Forcasted Amount for Next 1 year')
    pred_uc = results.get_forecast(steps=24)
    pred_ci = pred_uc.conf_int()
    ax = temp_df.plot(label='observed', figsize=(16, 8))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Order_Demand')
    plt.legend()
    plt.show()
    st.pyplot( )

else:
    st.text("Please Upload the File")