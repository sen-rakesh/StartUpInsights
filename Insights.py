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
st.markdown(""" ## Quick Analysis
Let's go!!  """)
raw_data = st.file_uploader("Upload a Dataset")
if raw_data is not None:
    data = pd.read_csv(raw_data,encoding='utf-8')
    st.dataframe(data.head(20))
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    col = ['Jan' , 'Feb' , 'Mar' , 'Apr' , 'May' , 'June' , 'July' , 'Aug' , 'Sep' , 'Oct' , 'Nov' , 'Dec']
    fund_year = [2015,2016,2017,2018,2019,2020]
    fund_df = pd.DataFrame(columns=col , index=fund_year)

    # Set the default values as zero
    fund_df[fund_df[::].isnull()] = 0

    fund_df_temp = pd.DataFrame(columns= ['Year' , 'Month' , 'Amount in USD'] )

    # Store the month wise funding recieved in the df.    
    for i,v in data.iterrows():
        mn = v['Date'].month
        yr = v['Date'].year
        fund_df.loc[yr][col[mn-1]] += v['Amount in USD']    
        fund_df_temp = fund_df_temp.append({'Year' : yr , 'Month' : mn , 'Amount in USD': v['Amount in USD'] } , ignore_index = True)
    
    # First we find out the monthly increase in data.

    if st.sidebar.checkbox("Monthly analysis of Funding Recieved"):
        sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
        ax = sns.relplot( height = 10,x = 'Month' , style = 'Year' ,kind = 'line', lw = 2, y = 'Amount in USD' , sort = col,  palette = ['black' , 'maroon' , 'navy' , 'limegreen' , 'mediumvioletred'], hue= 'Year', data = fund_df_temp)
        plt.xticks(np.arange(1,13) , col , rotation = 45)
        plt.title('Monthly analysis of Funding Recieved.')
        plt.show()
        st.pyplot()
    if st.sidebar.checkbox('Top Startups which recieved maximum funding'):
        num = int(st.sidebar.slider('Select number of top startups to show', 5, 20, 10))
        plt.figure(figsize=(12,12))
        sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
        sns.barplot(y = data.groupby('Startup Name').sum().sort_values(by = 'Amount in USD' ,ascending = False)[:num]['Amount in USD'].index , x = 'Amount in USD' , data = data.groupby('Startup Name').sum().sort_values(by = 'Amount in USD' ,ascending = False)[:num]) 
        plt.title('Top 20 Startups which recieved the maximum funding.')
        plt.show()
        st.pyplot()
    if st.sidebar.checkbox('Year-Wise Funding of startups'):
        plt.figure(figsize= (12,12))
        plt.ylabel('Amount in USD')         
        plt.xlabel('Year')
        sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
        sns.barplot(x = fund_df[::].index ,  y = [fund_df.loc[2015].values.sum() , fund_df.loc[2016].values.sum() , 
                fund_df.loc[2017].values.sum() , fund_df.loc[2018].values.sum() , fund_df.loc[2019].values.sum(),fund_df.loc[2020].values.sum()] , palette="RdBu" 
                ) 
        plt.title('Year-Wise Funding of startups.')
        plt.show()
        st.pyplot()
       
    if st.sidebar.checkbox('Top 20 Startups with largest no. of investors'):
        
        plt.figure(figsize=(12,12))
        plt.xticks(rotation = 90)
        sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
        sns.barplot(x = data.groupby('Startup Name').count().sort_values(by = 'No of Investors' ,ascending = False)[:20]['Amount in USD'].index , palette='GnBu' , y = 'No of Investors' , data = data.groupby('Startup Name').sum().sort_values(by = 'No of Investors' ,ascending = False)[:20]) 
        plt.title('Top 20 Startups with largest no of investors')
        plt.show()
        st.pyplot()
    
    if st.sidebar.checkbox('Find the investors who made the maximum investment'):
        st.markdown("### Investors who made the maximum investment")
        num2 = int(st.sidebar.slider('Select number of top investors to show', 5, 20, 10))
        data.groupby('Investors Name').sum().sort_values(by = 'No of Investors' ,ascending = False)[:num2]

    
    if st.sidebar.checkbox('Number of investments by each investor'):
        st.markdown("### Number of investments by each investor")
        investors_set = set()
        def seperate_investors(series):

            for i in series.values:
                if re.search(',' , i):
                    t_lst = i.split(',')
                    for j in t_lst:
                        investors_set.add(j)
                else:
                    investors_set.add(i)
                    
        seperate_investors(data['Investors Name'])

        # Now create a new dataframe.
        investment_df = pd.DataFrame(columns=['Investor Name' , 'No of Investment'])
        investment_df['No of Investment'] = investment_df['No of Investment'].astype('float') 
        # Initialize the dataframe.
        for i in investors_set:
            if i != '':
                investment_df = investment_df.append({'Investor Name':i , 'No of Investment':0} , ignore_index = True)
                
        # Populate the dataframe.
        for name in data['Investors Name']:
            if re.search(',', name):
                temp_lst = name.split(',')
                for nm in temp_lst:
                    investment_df.loc[investment_df['Investor Name'] == nm , 'No of Investment'] += 1.0 
            else:
                investment_df.loc[investment_df['Investor Name'] == name , 'No of Investment']  += 1.0 
                
        # investment_df[investment_df['Investor Name']== 'EQUITY CREST']['No of Investment'] = 3
        investment_df.loc[investment_df['Investor Name']== 'EQUITY CREST' , 'No of Investment'] += 1
        investment_df.sort_values(by= 'No of Investment', ascending=False)[:20]['Investor Name']
    
    if st.sidebar.checkbox('Top 20 Investors who invested the maximum number of times'):
        st.markdown("### Top 20 Investors who invested the maximum number of times")
        investors_set = set()
        def seperate_investors(series):

            for i in series.values:
                if re.search(',' , i):
                    t_lst = i.split(',')
                    for j in t_lst:
                        investors_set.add(j)
                else:
                    investors_set.add(i)
                    
        seperate_investors(data['Investors Name'])

        investment_df = pd.DataFrame(columns=['Investor Name' , 'No of Investment'])
        investment_df['No of Investment'] = investment_df['No of Investment'].astype('float') 
        # Initialize the dataframe.
        for i in investors_set:
            if i != '':
                investment_df = investment_df.append({'Investor Name':i , 'No of Investment':0} , ignore_index = True)
                
        # Populate the dataframe.
        for name in data['Investors Name']:
            if re.search(',', name):
                temp_lst = name.split(',')
                for nm in temp_lst:
                    investment_df.loc[investment_df['Investor Name'] == nm , 'No of Investment'] += 1.0 
            else:
                investment_df.loc[investment_df['Investor Name'] == name , 'No of Investment']  += 1.0 
                
        # investment_df[investment_df['Investor Name']== 'EQUITY CREST']['No of Investment'] = 3
        investment_df.loc[investment_df['Investor Name']== 'EQUITY CREST' , 'No of Investment'] += 1
        # Plot the graph
        plt.figure(figsize=(12,12))
        sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
        sns.barplot(y =investment_df.sort_values(by= 'No of Investment', ascending=False)[:20]['Investor Name'] , x = 'No of Investment' , data = investment_df.sort_values(by= 'No of Investment', ascending=False)[:20] , palette= 'dark') 
        plt.title('Top 20 Investors who invested the maximum number of times' )
        plt.show()
        st.pyplot()
    
    if st.sidebar.checkbox('Top 5 types of funding recieved by startups'):
        st.markdown("### Top 5 types of funding recieved by startups")
        # Here we find the type of funding(top 5) recieved by startups.
        font = {'family' : 'normal',
                'weight' : 'bold',
                'size'   : 15}
        plt.rc('font', **font)
        labels = data.groupby('InvestmentnType').sum().sort_values(by = 'Amount in USD' , ascending = False)[:5].index
        values = data.groupby('InvestmentnType').sum().sort_values(by = 'Amount in USD' , ascending = False)[:5]['Amount in USD'].values
        fig , ax = plt.subplots()
        fig.set_size_inches(12,12)
        ax.pie(colors = ['b' , 'g' , 'c' , 'm' , 'y'] ,  labels = labels , x = values , autopct='%.1f%%' , explode = [0.1 for x in range(5)])
        plt.title(' Top five types of funding recieved by startups.' , fontsize = 20)
        plt.show()
        st.pyplot()
    
    if st.sidebar.checkbox('Percentage of funding recieved by top ten industry verticals'):
        st.markdown("### Percentage of funding recieved by top ten industry verticals")
        # Here we find the industry vertical which recieved the maximum funding.
        font = {'family' : 'normal',
                'weight' : 'bold',
                'size'   : 22}
        plt.rc('font', **font)
        labels = data.groupby('Industry Vertical').sum().sort_values(by = 'Amount in USD' , ascending = False)[:10].index
        values = data.groupby('Industry Vertical').sum().sort_values(by = 'Amount in USD' , ascending = False)[:10]['Amount in USD'].values
        fig , ax = plt.subplots()
        fig.set_size_inches(12,12)
        ax.pie(  labels = labels , x = values , autopct='%.1f%%' , explode = [0.1 for x in range(10)])
        plt.title('Percentage of funding recieved by top ten industry verticals.' , fontsize = 30)
        plt.show()
        st.pyplot()
    
    if st.sidebar.checkbox('Top 10 Cities which recieved the maximum funding'):
        st.markdown("### Top 10 Cities which recieved the maximum funding")
        # Plot the data.
        city_data = data.groupby('City  Location' ).sum().sort_values(by = 'Amount in USD' , ascending = False)[:10]
        plt.figure(figsize=(12,12))
        sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
        sns.barplot(y = city_data.index , x = 'Amount in USD' , data = city_data , palette= 'pastel') 
        plt.title('Top 10 Cities which recieved the maximum funding.' )
        plt.show()
        st.pyplot()
    
    if st.sidebar.checkbox('Top 10 Cities which had the maximum number of investors'):
        st.markdown("### Top 10 Cities which had the maximum number of investors")
        # Plot the data.
        city_data = data.groupby('City  Location' ).sum().sort_values(by = 'No of Investors' , ascending = False)[:10]
        plt.figure(figsize=(12,12))
        sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
        sns.barplot(y = city_data.index , x = 'No of Investors' , data = city_data , palette= 'BuGn_r') 
        plt.title('Top 10 Cities which had the maximum number of investors.' )
        plt.show()
        st.pyplot()
    
    if st.sidebar.checkbox('Percentage of Top ten cities having maximum number of startup'):
        st.markdown("### Percentage of Top ten cities having maximum number of startup")
        top_cities = data.groupby('City  Location' ).count().sort_values(by = 'Startup Name' , ascending = False)[:10]
        # Here we find the industry vertical which recieved the maximum funding.
        font = {'family' : 'normal',
                'weight' : 'bold',
                'size'   : 15}
        plt.rc('font', **font)
        labels = top_cities.index
        values = top_cities['Startup Name'].values
        fig , ax = plt.subplots()
        fig.set_size_inches(12,12)
        ax.pie(colors = ['b' , 'g' , 'c' , 'm' , 'y'],labels = labels , x = values , autopct='%.1f%%' , explode = [0.1 for x in range(10)])
        plt.title('Percentage of Top ten cities having maximum number of startup' , fontsize = 30)
        plt.show()
        st.pyplot()
    
  

    




else:
    st.text("Please Upload the File")