#--------------------------------------------------------------------------
# Import Library
#--------------------------------------------------------------------------

#Import models from scikit learn module:
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

# for numpy array adn dataframe manipulation
import numpy as np
import pandas as pd

#--------------------------------------------------------------------------
# Define Functions
#--------------------------------------------------------------------------

#Generic function for making a classification model and accessing performance:
def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring,n_jobs=-1)
    return np.mean(xval)

#--------------------------------------------------------------------------
# Function for Random Forest Classifier and Gradient Boosting Classifier
#--------------------------------------------------------------------------
def classification_model_Test(model, X_train, y_train, X_test, y_test):
  #Fit the model:
  print("Fitting model ... \n")
  model.fit(X_train,y_train)
  
  print("*************************************************************************************")
  print("*       Home Sales Price Category Prediction Model Train/Test Results :             *")
  print("*************************************************************************************")

  print("Accuracy on training set: {:.3f}".format(model.score(X_train, y_train)))
  print("Accuracy on test set: {:.3f}".format(model.score(X_test, y_test)))  
    
  #Make predictions on training set:
  predictions = model.predict(X_test)
     
  # Get performance metrics
  cm = confusion_matrix(y_test,predictions)
  print(metrics.classification_report(y_test,predictions))
  score = compute_score(clf=model, X=X_test, y=y_test, scoring='accuracy')
  r2_score = metrics.r2_score(y_test,predictions)
  
  # Print performance metrics

  print ('*************************************************************')
  print ('Metric Performance of : {0}\n'.format(model.__class__))
  print('Confusion Matrix : \n', cm)
  print ('\nCross Validation Score = {0:.3%}\n'.format(score))
  print ("Coefficient of determination:{0:.3f}\n".format(r2_score))

#-----------------------------------------------------------------
# Function for Validation data testing
#-----------------------------------------------------------------
def classification_model_Val(model, X_val, y_val):
  print("*************************************************************************************")
  print("*       Home Sales Price Category Prediction Model Valuation Results :              *")
  print("*************************************************************************************")

  print("Accuracy on Valuation set: {:.3f}".format(model.score(X_val, y_val)))
  #Make predictions on training set:
  predictions = model.predict(X_val)
     
  # Get performance metrics
  cm = confusion_matrix(y_val,predictions)
  print(metrics.classification_report(y_val,predictions))
  score = compute_score(clf=model, X=X_val, y=y_val, scoring='accuracy')
  r2_score = metrics.r2_score(y_val,predictions)
  
  # Print performance metrics

  print ('*************************************************************')
  print ('Metric Performance of : {0}\n'.format(model.__class__))
  print('Confusion Matrix : \n', cm)
  print ('Cross Validation Score = {0:.3%}\n'.format(score))
  print ("Coefficient of determination:{0:.3f}\n".format(r2_score))

#--------------------------------------------------------------------------
# Checking on the feature importance weightage
#--------------------------------------------------------------------------
def features_importance(clf, data, predictors):
   
    features = pd.DataFrame()
    features['feature'] = predictors
    features['importance'] = clf.feature_importances_
    features.sort_values(by=['importance'], ascending=True, inplace=True)
    features.set_index('feature', inplace=True)
    print ('*************************************************************')
    features.plot(title = 'Features Importance',fontsize= 8, kind='barh', figsize=(50, 10))
    featimp = pd.Series(clf.feature_importances_, index=predictors).sort_values(ascending=False)
    print (featimp)
    print('\n**********************************************************')

def features_importance_model(clf, data, predictors):
 
    features = pd.DataFrame()
    features['feature'] = predictors
    features['importance'] = clf.feature_importances_
    features.sort_values(by=['importance'], ascending=True, inplace=True)
    features.set_index('feature', inplace=True)
    
    print('*************************************************************')
    print('*                    Feature Importance                     *')
    print('*************************************************************')

    featimp = pd.Series(clf.feature_importances_, index=predictors).sort_values(ascending=False)
    print (featimp)

#--------------------------------------------------------------------------
# Regression model test
#--------------------------------------------------------------------------
def Regression_model_Test(model, X_train, X_test, y_train, y_test):
    
    # Fit the model
    print("Fitting model ... \n")
    model.fit(X_train,y_train)

    print("*************************************************************************************")
    print("*       Home Sales Price Regression Prediction Model Train/Test Results :           *")
    print("*************************************************************************************\n")
 
    print("Accuracy on training set: {:.3f}".format(model.score(X_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(model.score(X_test, y_test)))
    
    # Model Prediction 
    y_predict = model.predict(X_test)

    # Scoring
    r2_score = metrics.r2_score(y_test,y_predict)    
    mae = mean_absolute_error(y_test.values.ravel(), y_predict)  # Mean absolute error (MAE)
    mse = mean_squared_error(y_test.values.ravel(), y_predict)   # Mean squared error (MSE)
    rmse = np.sqrt(mse)

    # Performance metrics
    print('*************************************************************')
    print("MAE:{0:.3f}".format(mae))
    print("MSE:{0:.3f}".format(mse))
    print("RMSE:{0:.3f}".format(rmse))
    print("R2 Score:{0:.3f}".format(r2_score))
    print('*************************************************************\n')


def Regression_model_Val(model, X_val, y_val):

    print("*************************************************************************************")
    print("*       Home Sales Price Regression Prediction Model Valuation Results :            *")
    print("*************************************************************************************\n")  

    print("Accuracy on val set: {:.3f}".format(model.score(X_val, y_val)))
       
    # Model Prediction 
    y_predict = model.predict(X_val)

    # Scoring
    score = compute_score(clf=model, X=X_val, y=y_val, scoring='neg_mean_absolute_error')
    r2_score = metrics.r2_score(y_val,y_predict)    
    mae = mean_absolute_error(y_val.values.ravel(), y_predict)  # Mean absolute error (MAE)
    mse = mean_squared_error(y_val.values.ravel(), y_predict)   # Mean squared error (MSE)
    rmse = np.sqrt(mse)

    # Performance metrics
    print('*************************************************************')
    print("MAE:{0:.3f}".format(mae))
    print("MSE:{0:.3f}".format(mse))
    print("RMSE:{0:.3f}".format(rmse))
    print("R2 Score:{0:.3f}".format(r2_score))
    print('*************************************************************\n')