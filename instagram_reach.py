## Importing Libraries and Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.dates import DateFormatter
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

Insta_data = pd.read_csv('Instagram-Reach.csv')

print(Insta_data.head())


## Q.1: Import data and check null values, column info, and descriptive statistics of the data.

# Null values
null_data = Insta_data.isnull().sum()

# Descriptive statistics
descriptive_stats = Insta_data.describe()

print("Null values in the dataset:\n", null_data)

print("Column info:\n")
Insta_data.info()

print("Descriptive statistics:\n", descriptive_stats)


# Now Converting Date column to datetime
Insta_data['Date'] = pd.to_datetime(Insta_data['Date'])

print(data.info())

## Q.3: Analyze the trend of Instagram reach over time using a line chart.

plt.figure(figsize=(12, 6))
plt.plot(Insta_data['Date'], Insta_data['Instagram reach'], label='Instagram reach')
plt.xlabel('Date')
plt.ylabel('Reach')
plt.title('Instagram Reach Over Time')
plt.legend()
plt.grid(True)
plt.show()


## Q.4: Analyze Instagram reach for each day using a bar chart.

plt.figure(figsize=(12, 6))
plt.bar(Insta_data['Date'], Insta_data['Instagram reach'], color='blue')
plt.title('Instagram Reach for Each Day')
plt.xlabel('Date')
plt.ylabel('Instagram Reach')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


## Q.5: Analyze the distribution of Instagram reach using a box plot.

plt.figure(figsize=(12, 6))
sns.boxplot(x=Insta_data['Instagram reach'])
plt.xlabel('Reach')
plt.title('Distribution of Instagram Reach')
plt.show()


## Q.6: Create a day column and analyze reach based on the days of the week.

# Setting Date column as the index
Insta_data.set_index('Date', inplace=True)

# Creating a day column
Insta_data['Day'] = Insta_data.index.day_name()
print(Insta_data.head())

# Now Grouping by Day and calculating mean, median, standard deviation
day_grouped = Insta_data.groupby('Day')['Instagram reach'].agg(['mean', 'median', 'std']).reset_index()

print(day_grouped)

## Q.7: Create a bar chart to visualize the reach for each day of the week.

plt.figure(figsize=(10, 6))
sns.barplot(x='Day', y='mean', Insta_data=day_grouped)
plt.xlabel('Day of the Week')
plt.ylabel('Mean Reach')
plt.title('Average Instagram Reach by Day of the Week')
plt.show()

## Q.8: Check the Trends and Seasonal patterns of Instagram reach


result = seasonal_decompose(Insta_data['Instagram reach'], model='additive', period=30)  
# Plotting the decomposition
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12))
result.observed.plot(ax=ax1)
result.trend.plot(ax=ax2)
result.seasonal.plot(ax=ax3)
result.resid.plot(ax=ax4)

# Setting titles
ax1.set_title('Instagram Reach')
ax2.set_title('Trend')
ax3.set_title('Seasonal')
ax4.set_title('Residual')

# Setting date format on x-axis
date_form = DateFormatter("%Y-%m")
ax1.xaxis.set_major_formatter(date_form)
ax2.xaxis.set_major_formatter(date_form)
ax3.xaxis.set_major_formatter(date_form)
ax4.xaxis.set_major_formatter(date_form)

# Rotating date labels
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.show()

##  Q9: Use the SARIMA model to forecast the reach of the Instagram account

# Gven value of d
d = 1

# Determine p using ACF plot
plot_acf(Insta_data['Instagram reach'],)
plt.show()

# Determine q using PACF plot
plot_pacf(Insta_data['Instagram reach'],)
plt.show()


## Q10: Train a model using SARIMA and make predictions

# Defining the SARIMA model
p = 1 # From PACF plot
q = 1 # Rrom ACF plot
model = SARIMAX(Insta_data['Instagram reach'], order=(p, d, q), seasonal_order=(p, d, q, 12))

# Fitting the model
model_fit = model.fit(disp=False)

predictions = model_fit.predict(start=len(Insta_data), end=len(Insta_data) + 30, typ='levels')
print(predictions)

plt.figure(figsize=(12, 6))
plt.plot(Insta_data['Instagram reach'], label='Historical')
plt.plot(predictions, label='Forecast', color='red')
plt.title('Instagram Reach Forecast')
plt.xlabel('Date')
plt.ylabel('Instagram Reach')
plt.legend()
plt.show()

