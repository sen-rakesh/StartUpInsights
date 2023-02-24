import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline

#importing statmodels
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#importing metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, auc, roc_curve

#importing sklearn-mdoels
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import f_regression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error

#importing skabslearn-mdoels
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')

import warnings
warnings.filterwarnings("ignore")
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title = "Bitathon 2023")
st.markdown(""" ## Profit Prediction using Random Forest
""")
raw_data = st.file_uploader("Upload a Dataset")
if raw_data is not None:
    data2 = pd.read_csv(raw_data,encoding='utf-8')
    one_hot_encode = pd.get_dummies(data2.State)
    data2.drop(['State'], axis=1, inplace=True)
    Q1 = data2.Profit.quantile(0.25)
    Q3 = data2.Profit.quantile(0.75)
    IQR = Q3 - Q1
    data2 = data2[(data2.Profit >= Q1 - 1.5*IQR) & (data2.Profit <= Q3 + 1.5*IQR)]
    st.subheader('Pair Plot')
    sns.pairplot(data2[['R&D Expenditure', 'Administration Expenses', 'Marketing Expenditure', 'Profit']], kind='reg',diag_kind='kde')
    plt.show()
    st.pyplot()
    st.subheader('Correlation Heatmap')
    data2.drop(['Funding Rounds','Initial Seed Funding (In Crores)','SISFS','International Sales','Primary Owner- Female '], axis=1, inplace=True)
    sns.heatmap(data2.corr(), annot=True)
    st.pyplot()
    st.subheader('Joint Plot R&D Expenditure vs Profit')
    sns.jointplot(x=data2['Profit'],y=data2['R&D Expenditure'], kind='reg')
    plt.show()
    st.pyplot()
    dataset_prepared=data2.copy()
    numerical = dataset_prepared.drop(columns=['Profit'])
    vif=pd.DataFrame()
    vif['Features']=numerical.columns
    vif['VIF']=[variance_inflation_factor(numerical.values,i) for i in range(numerical.shape[1])]
    vif['VIF']=round(vif['VIF'],2)
    vif=vif.sort_values(by='VIF',ascending=False)

    dataset_prepared=pd.get_dummies(dataset_prepared,drop_first=True)
    dataset_prepared.rename(columns={'R&D Expenditure':"R&D",'Marketing Expenditure':'Marketing'},inplace=True)
    dataset_prepared = pd.concat([dataset_prepared, one_hot_encode], axis=1)
    dataset_prepared.dropna(inplace=True,axis=0)

    x= dataset_prepared.drop(columns='Profit')
    y= dataset_prepared.Profit

    data2=f_regression(x[['R&D','Administration Expenses','Marketing']],y)
    f_df=pd.DataFrame(data2,index=[['F_statistic','p_value']],columns=x[['R&D','Administration Expenses','Marketing']].columns).T

    x=x.drop(columns="Administration Expenses")

    x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.1, random_state=6)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_percentage_error,r2_score

    model2=RandomForestRegressor(n_estimators=500,max_depth=17)

    model2.fit(x_train,y_train)
    pred=model2.predict(x_test)
    mape=mean_absolute_percentage_error(pred,y_test)
    r2=r2_score(pred,y_test)
    st.subheader(f'Accuracy is {round(r2*100,2)}%')
    st.dataframe(pd.DataFrame({'MAPE': [mape], 'R2 Score': [r2]}))
    st.subheader('Actual Vs Predicted Profit')
    df = pd.DataFrame({'Actual': y_test, 'Predicted': pred})

    st.dataframe(df)

else:
    st.text("Please Upload the File")
