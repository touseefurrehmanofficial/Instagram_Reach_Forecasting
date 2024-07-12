# Instagram_Reach_Forecasting
This repository provides a structured approach to analyzing and forecasting Instagram reach using historical data. The primary goal is to identify patterns, trends, seasonality, and anomalies to develop a predictive model capable of forecasting future reach. The analysis and forecasting are implemented using Python and several data analysis libraries.

## Dataset

The dataset, `Instagram-Reach.csv`, includes:
- **Date**: The date of the Instagram post.
- **Instagram Reach**: The number of people reached by the post on the corresponding date.

## Project Structure

- **data/**: Contains the dataset.
- **notebooks/**: Jupyter notebooks with analysis and forecasting steps.
- **scripts/**: Python scripts for data analysis and modeling.
- **results/**: Output plots and forecast results.

## Analysis and Forecasting Steps

### 1. Import Data and Initial Checks
- Import the dataset.
- Check for null values, column info, and descriptive statistics.

### 2. Convert Date Column to Datetime
- Convert `Date` to datetime datatype.
- Set `Date` as the index.

### 3. Trend Analysis
- Plot a line chart to visualize Instagram reach trends over time.

### 4. Daily Reach Analysis
- Plot a bar chart for daily Instagram reach.

### 5. Distribution Analysis
- Create a box plot to visualize Instagram reach distribution.

### 6. Weekly Reach Analysis
- Create a `Day` column from the `Date` column.
- Group by day and calculate mean, median, and standard deviation of Instagram reach.

### 7. Weekly Reach Visualization
- Plot a bar chart for mean Instagram reach by day of the week.

### 8. Seasonal Decomposition
- Decompose the time series to analyze trends and seasonal patterns.

### 9. SARIMA Model Parameter Determination
- Use autocorrelation and partial autocorrelation plots to determine SARIMA parameters (p, d, q).

### 10. SARIMA Model Training and Forecasting
- Train a SARIMA model and make predictions.
- Plot historical data and forecasted values.

## Requirements

Install required libraries using:
```bash
pip install pandas numpy matplotlib seaborn statsmodels
```

