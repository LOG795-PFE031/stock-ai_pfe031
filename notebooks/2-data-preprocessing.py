#!/usr/bin/env python
# coding: utf-8

# # LSTM Time Series - Stock Price Prediction
# ## Part 2 - Data Preprocessing
# This notebook focuses on processing the filtered dataset containing the historical prices of Google stocks over the past five years.
# 
# > **INPUT**: Filtered dataset containing Google's stock prices from the last five years, obtained from the previous phase. <br/>
# > **OUTPUT**: Preprocessed and transformed data divided into training, validation, and testing subsets, stored in an interim location for the training phase.

# ### 1. INITIALIZATION

# In[1]:


# Import necessary libraries and modules
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib
import matplotlib.dates as mdates


# In[2]:


# Set output width
pd.set_option("display.width", 120)


# ### 2. LOADING DATASET

# In[16]:


# Prepare data file location and name
data_file_location = "..//data//interim//"
data_file_name = "2025_google_stock_price_recent"
data_file_ext = "csv"

# Load data file
data = pd.read_csv(data_file_location + data_file_name + "." + data_file_ext)


# In[17]:


# Check loaded data shape
data.shape


# In[20]:


# Check loaded data head
data.head()


# In[21]:


# Check columns types
data.dtypes


# ### 3. DATA PREPROCESSING

# #### Validate Data Types

# In[22]:


# Convert date column to a valid Datetime format
data["Date"] = pd.to_datetime(data["Date"])

# Check column types
data.dtypes


# #### Select Independent Features

# The objective of this analysis is to implement a multi-variant prediction, taking into account possible impact of several independent features such as the Open price, Close price and Volume on future stock price performance.
# 
# Therefore, in this analysis, we will incorporate all the available variables: 
# - Opening price
# - Highest price
# - Lowest price
# - Closing price
# - Adjusted closing price
# - Trading volume
# 
# These features will be utilized to forecast the future opening price.

# In[23]:


# Define selected features and target attribute
features = ["Open", "High", "Low",	"Close", "Adj Close", "Volume"]
target = "Open"


# #### Create Train, Validation, and Test Datasets

# To monitor and assess the performance of our model, we will partition the recent stock price dataset into three segments: training, validation, and testing sets.
# 
# The division will be structured as follows:
# - **Training dataset:** covering the period from the start of 2019 till June, 2023.
# - **Validation dataset:** representing the stock prices from July, 2023 till the end of 2023.
# - **Testing dataset:** representing the stock prices for the first two months of 2024.

# In[25]:


# Define start and end time for each period
train_end_date = pd.to_datetime("2024-05-30")
validate_start_date = pd.to_datetime("2024-06-01")
validate_end_date = pd.to_datetime("2024-12-31")
test_start_date = pd.to_datetime("2025-01-01")
test_end_date = pd.to_datetime("2025-02-09")

# Split dataset into training, validation, and testing
data_train = data[data["Date"] <= train_end_date][features]
data_train_dates = data[data["Date"] <= train_end_date]["Date"]
data_validate = data[(data["Date"] >= validate_start_date) & (data["Date"] <= validate_end_date)][features]
data_validate_dates = data[(data["Date"] >= validate_start_date) & (data["Date"] <= validate_end_date)]["Date"]
data_test = data[(data["Date"] >= test_start_date) & (data["Date"] <= test_end_date)][features]
data_test_dates = data[(data["Date"] >= test_start_date) & (data["Date"] <= test_end_date)]["Date"]


# In[26]:


# Display the shape of each dataset
print(f"Training Set: {data_train.shape}")
print(f"Validation Set: {data_validate.shape}")
print(f"Testing Set: {data_test.shape}")


# In[27]:


# Display a summary of each dataset
print("Training Dataset:")
print(data_train.head())
print("Validation Dataset:")
print(data_validate.head())
print("Testing Dataset:")
print(data_test.head())


# In[28]:


# Plot stock prices for each data split
plt.figure(figsize=(18,6))
plt.plot(data_train_dates, data_train["Open"], color="cornflowerblue")
plt.plot(data_validate_dates, data_validate["Open"], color="orange")
plt.plot(data_test_dates, data_test["Open"], color="green")
plt.legend(["Train Data", "Validation Data", "Test Data"])
plt.title("Data Split for Google Stock Price")
plt.xlabel("Samples Over Time")
plt.ylabel("Price (USD)")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
plt.grid()


# #### Data Scaling

# In[29]:


# Check the distribution of input features
data[features].describe()


# Looking at these details, it's noticeable and anticipated that all price values exhibit similar distributions given they log the price fluctuation on daily basis.
# 
# Conversely, the trading volume presents a distinct distribution that differs significantly.
# 
# Accordingly, input features need to be transformed into a unified scale and since the distribution doesn't indicate any outliers, we will use the [0,1] range to normalize all features.
# 
# To prevent data leakage, we will fit the scaler solely to the training data. Subsequently, we will use this fitted scaler to transform the training, validation, and testing datasets.

# In[30]:


# Initialize scaler with range [0,1]
sc = MinMaxScaler(feature_range=(0,1))

# Fit and transform scaler to training set
data_train_scaled = sc.fit_transform(data_train)

# Transform validating and testing datasets
data_validate_scaled = sc.transform(data_validate)
data_test_scaled = sc.transform(data_test)


# The scaler employed here will also be utilized in subsequent phases to revert the scaled data back to its original distribution. Therefore, it is essential to save this scaler to a local folder for future use.

# In[31]:


# Prepare scaler model name and location
scaler_model_location = "..//models//"
scaler_model_name = "2025_google_stock_price_scaler"
scaler_model_ext = "gz"

# Store scaler model
joblib.dump(sc, scaler_model_location + scaler_model_name + "." + scaler_model_ext)


# ### 4. STORING PROCESSED DATASETS

# The training, validation, and testing datasets have been processed and are prepared for training the LSTM model in the next phase.
# 
# Prior to saving these datasets, it is necessary to reassemble the dates corresponding to each dataset. This will facilitate later evaluation of the model's performance.

# In[32]:


# Combine dates with each corresponding dataset
data_train_scaled_final = pd.DataFrame(data_train_scaled, columns=features, index=None)
data_train_scaled_final["Date"] = data_train_dates.values

data_validate_scaled_final = pd.DataFrame(data_validate_scaled, columns=features, index=None)
data_validate_scaled_final["Date"] = data_validate_dates.values

data_test_scaled_final = pd.DataFrame(data_test_scaled, columns=features, index=None)
data_test_scaled_final["Date"] = data_test_dates.values


# In[33]:


# Prepare datasets files and location
data_file_location = "..//data//processed//"
data_file_name_train = "2025_google_stock_price_processed_train"
data_file_name_validate = "2025_google_stock_price_processed_validate"
data_file_name_test = "2025_google_stock_price_processed_test"
data_file_ext = "csv"

# Store datasets
data_train_scaled_final.to_csv(data_file_location + data_file_name_train + "." + data_file_ext, index=None)
data_validate_scaled_final.to_csv(data_file_location + data_file_name_validate + "." + data_file_ext, index=None)
data_test_scaled_final.to_csv(data_file_location + data_file_name_test + "." + data_file_ext, index=None)

