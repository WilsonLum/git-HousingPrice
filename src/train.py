#!/usr/bin/env python

#*************************************************************
# Import Library
#*************************************************************

# for path manipulation
import os

# For pandas dataframe and numpy array library 
import pandas as pd
import numpy as np

# For SQLite library
import sql_tools

# For label encoding of data
from sklearn.preprocessing import LabelEncoder

# Import from src library
import ML_metric

# Machine Learning Model library import
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# import warnings filter
from warnings import simplefilter
import warnings

warnings.filterwarnings("ignore")

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

#*************************************************************
# Define Database Path
#*************************************************************

path=os.getcwd()
path=os.path.split(path)

database = path[0] +"\data/home_sales.db"

#*************************************************************
# Reading in the dataset
#*************************************************************

# create a database connection
conn = sql_tools.create_connection(database)

sql ='SELECT * FROM sales'
df = pd.read_sql(sql, con=conn)

#*************************************************************
# Preprocessed data & Feature Engineering
#*************************************************************

# Turn all to lower case for 'condition' & get year feature from 'date'
df['condition'] = df.condition.str.lower()
drop_df = df.copy()
drop_df['year'] = drop_df['date'].str[-4:]

# Drop unuseful features and null rows
columns = ['built', 'longitude','latitude','date','lot_size','price']
drop_df = drop_df.replace(0, pd.np.nan).dropna(axis=0, how='any', subset=columns).fillna(0)
drop_df = df.drop(['id','basement_size','renovation','date'],axis=1)
drop_df = drop_df.dropna()

#*************************************************************
# Category Data Encoding
#*************************************************************

cat_features = ['bedrooms','bathrooms','review_score','condition']

le = LabelEncoder()
for i in cat_features:
    drop_df[i] = le.fit_transform(drop_df[i].astype(str))

#*************************************************************
# Create dataset for regression
#*************************************************************

regression_df = drop_df.copy()

X = regression_df.drop(['price'], axis=1)
y = regression_df['price']

#*************************************************************
# Prepare train-val-test classification dataset
#*************************************************************
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8,shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=8,shuffle=True)

outcome_var = 'price'
predictor_var = X.columns


#*************************************************************
# RFC Model Training
#*************************************************************

RFR_model = RandomForestRegressor(max_depth = 60,max_features = 'sqrt', n_estimators = 500,random_state=8)
ML_metric.Regression_model_Test(RFR_model, X_train, X_test, y_train, y_test)


#*************************************************************
# Model Test results
#*************************************************************

# Load the classifier model
ML_metric.Regression_model_Val(RFR_model, X_val, y_val)

# Print out the feature importance
ML_metric.features_importance_model(RFR_model,regression_df,predictor_var)
