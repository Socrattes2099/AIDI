## DATASET FROM KAGGLE: https://www.kaggle.com/harlfoxem/housesalesprediction

# ✔️Date: Date house was sold
# ✔️Price: Price is prediction target
# ✔️Bedrooms: Number of Bedrooms/House
# ✔️Bathrooms: Number of bathrooms/House
# ✔️Sqft_Living: square footage of the home
# ✔️Sqft_Lot: square footage of the lot
# ✔️Floors: Total floors (levels) in house
# ✔️Waterfront: House which has a view to a waterfront
# ✔️View: Has been viewed
# ✔️Condition: How good the condition is ( Overall )
# ✔️Grade: grade given to the housing unit, based on King County grading system
# ✔️Sqft_Above: square footage of house apart from basement
# ✔️Sqft_Basement: square footage of the basement
# ✔️Yr_Built: Built Year
# ✔️Yr_Renovated: Year when house was renovated
# ✔️Zipcode: Zip
# ✔️Lat: Latitude coordinate
# ✔️Long: Longitude coordinate
# ✔️Sqft_Living15: Living room area in 2015(implies — some renovations)
# ✔️Sqft_Lot15: lotSize area in 2015(implies — some renovations)

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
from numpy import cov
# calculate the Pearson's correlation between two variables
from scipy.stats import pearsonr
# %matplotlib inline

def calculate_correlation(data1, data2):
  # calculate Pearson's correlation
  corr, _ = pearsonr(data1, data2)
  print("Pearson's correlation: %.3f" % corr)

  return corr

data = pd.read_csv("kc_house_data.csv")

data.head()

data.describe()

# Most common house per number of bedrooms
data['bedrooms'].value_counts().plot(kind='bar')
plt.title('number of Bedroom')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
sns.despine

# Location of houses based on latitude and longitude
plt.figure(figsize=(10,10))
sns.jointplot(x=data.lat.values, y=data.long.values, height=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()
sns.despine

# Correlation between Price and Grade
plt.figure(figsize=(10, 10))
plt.scatter(data.price,data.grade)
plt.xlabel("Price")
plt.ylabel("Grade")
plt.title("Price vs Grade")

calculate_correlation(data.price, data.grade)

# Correlation between Price and Area
plt.figure(figsize=(10, 10))
plt.scatter(data.price,data.sqft_living)
plt.xlabel("Price")
plt.ylabel("Square Feet (ft^2)")
plt.title("Price vs Square Feet")

calculate_correlation(data.price, data.sqft_living)

# Correlation between Price and Geographical Longitude
plt.figure(figsize=(10, 10))
plt.scatter(data.price,data.long)
plt.xlabel("Price")
plt.ylabel("Longitude")
plt.title("Price vs Longitude (location coordinate)")

calculate_correlation(data.price, data.long)

# Correlation between Price and Geographical Latitude
plt.figure(figsize=(10, 10))
plt.scatter(data.price,data.lat)
plt.xlabel("Price")
plt.ylabel('Latitude')
plt.title("Price vs Latitude (location coordinate)")

calculate_correlation(data.price, data.lat)

# Correlation between Price and Number of Bedrooms
plt.figure(figsize=(10, 10))
plt.scatter(data.bedrooms,data.price)
plt.title("Number of Bedroom and Price ")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.show()
sns.despine

calculate_correlation(data.bedrooms,data.price)

# Correlation between Price and Bedrooms after removing outlier
data = data.drop(data.bedrooms.idxmax()) # Remove outlier of 30 bedrooms
plt.figure(figsize=(10, 10))
plt.scatter(data.bedrooms,data.price)
plt.title("Number of Bedroom and Price ")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.show()
sns.despine

calculate_correlation(data.bedrooms,data.price)

# Correlation between Area of Living Room and Basement
plt.figure(figsize=(10, 10))
plt.scatter((data['sqft_living']+data['sqft_basement']),data['price'])
plt.xlabel("ft^2 living room")
plt.ylabel("ft^2 basement")
plt.title("Square feet Living room vs Square feet Basement")

calculate_correlation(data['sqft_living']+data['sqft_basement'],data['price'])

# Correlation between Price and Existence of Waterfront
plt.figure(figsize=(10, 10))
plt.scatter(data.waterfront,data.price)
plt.title("Waterfront vs Price")
plt.xticks(np.arange(2), ['No', 'Yes'])
plt.xlabel("Waterfront? (Yes/No)")
plt.ylabel("Price")

calculate_correlation(data.waterfront,data.price)

train1 = data.drop(['id', 'price'],axis=1)

train1.head()

# Frecuency of different number of floors
plt.figure(figsize=(10, 10))
data.floors.value_counts().plot(kind='bar')
plt.title('Number of Floors')
plt.xlabel('Floors')
plt.ylabel('Count')
sns.despine

# Correlation between Price and Number of Floors
plt.figure(figsize=(10, 10))
plt.scatter(data.floors,data.price)
plt.xlabel("# of Floors")
plt.ylabel("Price")
plt.title("Number of Floors vs Price")

calculate_correlation(data.floors,data.price)

# Correlation between Price and House Condition ranked from 1 to 5
plt.figure(figsize=(10, 10))
plt.scatter(data.condition,data.price)
plt.xlabel("Condition rating (1=worst, 5=best)")
plt.ylabel("Price")
plt.title("Condition vs Price")

calculate_correlation(data.condition,data.price)

# Correlation between Price and Zipcode
plt.figure(figsize=(10, 10))
plt.scatter(data.zipcode,data.price)
plt.xlabel("Zipcode")
plt.ylabel("Price")
plt.title("Which is the pricey location by zipcode?")

calculate_correlation(data.condition,data.price)

from sklearn.linear_model import LinearRegression
from sklearn import metrics

print("\n## Training and Testing LinearRegressor ##")
reg = LinearRegression()

from  datetime import datetime

labels = data['price']
# Separated houses older than 2015 from newer houses, by assigning a 0 and 1 value respectively
conv_dates = [0 if int(value[:4]) < 2015 else 1 for value in data.date ]
data['date'] = conv_dates

# Drop id and price because we don't need it for the training dataset
train1 = data.drop(['id', 'price'],axis=1)

from sklearn.model_selection import train_test_split
# Split data into training and testing sets
x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.10,random_state =2)

from time import time
t=time()

reg.fit(x_train,y_train)
print("training time=",time()-t)

from time import time
t=time()
print("score is",reg.score(x_test,y_test))
print("testing time=",time()-t)

### Use the Gradient Boosting ensemble regressor to improve the accuracy of our predictions ###
# Parameters:
# n_estimator — The number of boosting stages to perform. n=100 is enough as the error reaches a stable amount and also the model is overfit.
# max_depth — The depth of the tree node.
# learning_rate — Rate of learning the data.
# loss — loss function to be optimized. ‘ls’ refers to least squares regression

from sklearn import ensemble
print("\n## Training and Testing GradientBoostingRegressor ##")
params = {}
params['n_estimators'] = 100
clf = ensemble.GradientBoostingRegressor(n_estimators = params['n_estimators'], max_depth = 5, min_samples_split = 2,
          learning_rate = 0.1, loss = 'ls')

t=time()
clf.fit(x_train, y_train)
print("training time=",time()-t)

t=time()
print("score is",clf.score(x_test,y_test))
print("testing time=",time()-t)

# Plotting Deviance/Error Chart
test_score = np.zeros((params['n_estimators']),dtype=np.float64)

for i,y_pred in enumerate(clf.staged_predict(x_test)):
    test_score[i]=clf.loss_(y_test,y_pred)

plt.figure(figsize=(18, 9))
plt.subplot(1, 2, 1)
plt.plot(np.arange(params['n_estimators']) + 1,clf.train_score_,'b-',label= 'Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1,test_score,'r-',label = 'Test Set Deviance')
plt.xlabel("Boosting Iterations")
plt.ylabel("Deviance")
plt.title('Deviance (Loss)')
plt.legend()

import sklearn.inspection as inspection

# Plot Feature Importance and Permutation Importance
feature_importance = clf.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(train1.columns)[sorted_idx])
plt.title('Feature Importance (MDI)')

result = inspection.permutation_importance(clf, x_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=2)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(result.importances[sorted_idx].T,
            vert=False, labels=np.array(train1.columns)[sorted_idx])
plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.show()

# Function to display portion of dataset and display model metrics
# Params: regressor and number of samples to predict 
def show_regressor_predictions(regressor, n_pred = 15):

  # Predictions
  df_test = x_test[:n_pred] # Extract samples from test dataset
  y_pred = regressor.predict(df_test) # Predict using Linear Regressor
  y_actual = y_test[:n_pred]  # Extract actual values
  diff_prices = (y_pred - y_actual)/y_actual # Calculate difference
  df_test.index = np.arange(1, n_pred+1)

  print("\n## Showing predictions for regressor:", regressor.__class__.__name__, "###\n")

  # Append columns for Actual and Predicted Price, and difference
  df_test.insert(len(df_test.columns), "Actual Price", ["${:,.2f}". format(price) for price in y_actual], True)
  df_test.insert(len(df_test.columns), "Predicted Price", ["${:,.2f}". format(price) for price in y_pred], True)
  df_test.insert(len(df_test.columns), "Diff (%)", ["{:,.2f}%". format(diff * 100) for diff in diff_prices], True)

  # Output features and predictions
  display(df_test)

  # Calculate Mean absolute percentage error
  absolute_error = metrics.mean_absolute_error(y_actual, y_pred)
  print("\nAbsolute error is: ", "${:,.2f}".format(absolute_error))
  # Calculate Coefficient of determination (R-squared)
  r_squared = metrics.r2_score(y_actual, y_pred)
  print("Coefficient of determination is: ", "{:,.2f}".format(r_squared))
  print("\n----------------------------------------------\n\n")

show_regressor_predictions(reg)
show_regressor_predictions(clf)
