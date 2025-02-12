#!/usr/bin/env python
# coding: utf-8

# # LSTM Time Series - Stock Price Prediction
# ## Part 1 - Data Explanatory Analysis
# This notebook focuses on examining the raw dataset containing daily historical prices of Google stocks. The goal is to identify a specific timeframe suitable for further analysis and prediction using LSTM.
# 
# > **INPUT**: The raw data file of all available stock prices of Google (Alphabet Inc.), as downloaded from its original source. <br/>
# > **OUTPUT**: The extracted historical data of the targeted period for analysis, stored in an intermediary location for next steps.

# ### 1. INITIALIZATION

# In[1]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt


# ### 2. LOAD DATASET FILE

# The dataset we are going to initially load and explore is the historical records of daily price details of Google (Alphabet Inc.) sourced from [Yahoo Finance](https://finance.yahoo.com/quote/GOOG).
# 
# At first, we load the complete dataset available spanning a period from 2004-08-19 up until the time of creating this script.
# 
# The main objective of loading the entire dataset is to choose a particular time frame for analysis.

# In[2]:


# Prepare data file location and load the dataset
data_file_location = "../data/raw/"
data_file_name = "2025_google_stock_price_full"
data_file_ext = "csv"

# Load data file
data = pd.read_csv(data_file_location + data_file_name + "." + data_file_ext)


# In[3]:


# Check dataset shape
data.shape


# In[6]:


# Check dataset head
data.head()


# In[12]:


# Check data types
data.dtypes


# ### 3. INITIAL DATA CLEANING

# As we notice, the Date column is currently in a String format which requires conversion to proper Datetime format.

# In[13]:


# Convert Date column to a valid Datetime format
data["Date"] = pd.to_datetime(data["Date"])


# In[9]:


# Check column format
data.dtypes


# ### 4. EXPLORE DATASET

# In[14]:


# Plot Open and Close price for the whole period
plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
plt.plot(data["Date"], data["Open"])
plt.xlabel("Time")
plt.ylabel("Open Price (USD)")
plt.title("Google Open Stock Price")
plt.grid()

plt.subplot(1,2,2)
plt.plot(data["Date"], data["Close"])
plt.xlabel("Time")
plt.ylabel("Close Price (USD)")
plt.title("Google Close Stock Price")
plt.grid()

plt.suptitle("Google Stock Price Over Time")
plt.show()


# The previous charts show an inconsistent behavior in the stock's performance over time, with a completely distinct trend during the past five years.
# 
# This means that the old historical data might not be relevant to the predictions, since it's most likely to represents a different period with outdated indicators that don't affect the current trend.
# 
# Accordingly, we will concentrate our analysis on the most recent five-year data, as it is expected to provide more accurate insights for predicting future trends.

# In[15]:


# Select stock price records for the last five years, starting from 2019
data_5years = data[data["Date"].dt.year >= 2019]

# Check filtered data shape
data_5years.shape


# In[16]:


# Plot Open stock price performance in the last five years
plt.figure(figsize=(18,6))
plt.plot(data_5years["Date"], data_5years["Open"])
plt.xlabel("Time")
plt.ylabel("Stock Price (USD)")
plt.title("Google Open Stock Price - Starting From 2019")
plt.grid()


# The selected period seems to be more relevant for analysis, as it's more likely to drive the stock performance with the impact of other external factors.

# ### 5. STORE FILTERED DATASET

# In[17]:


# Prepare data file location
data_file_location = "..//data//interim//"
data_file_name = "2025_google_stock_price_recent"
data_file_ext = "csv"

# Store dataset
data_5years.to_csv(data_file_location + data_file_name + "." + data_file_ext, index=None)

