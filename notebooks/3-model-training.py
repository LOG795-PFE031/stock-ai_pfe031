#!/usr/bin/env python
# coding: utf-8

# # LSTM Time Series - Stock Price Prediction
# ## Part 3 - Model Training
# In this notebook, we import the scaled dataset files, prepare them in a format suitable for LSTM modeling, and proceed to train the LSTM model.
# 
# > **INPUT**: Scaled dataset files for training, validation, and testing periods, as processed in the preceding phase. <br/>
# > **OUTPUT**: Trained LSTM model and analysis of its performance.

# ### 1. INITIALIZATION

# In[1]:


# Import necessary libraries and modules
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
import joblib
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


# ### 2. LOADING DATASETS

# In[3]:


# Prepare data file location and name
data_file_location = "..//data//processed//"
data_file_name_train = "2025_google_stock_price_processed_train"
data_file_name_validate = "2025_google_stock_price_processed_validate"
data_file_name_test = "2025_google_stock_price_processed_test"
data_file_ext = "csv"

# Load data files
data_train_df = pd.read_csv(data_file_location + data_file_name_train + "." + data_file_ext)
data_validate_df = pd.read_csv(data_file_location + data_file_name_validate + "." + data_file_ext)
data_test_df = pd.read_csv(data_file_location + data_file_name_test + "." + data_file_ext)


# In[4]:


# Check loaded datasets shape
print(f"Training Dataset Shape: {data_train_df.shape}")
print(f"Validation Dataset Shape: {data_validate_df.shape}")
print(f"Testing Dataset Shape: {data_test_df.shape}")


# In[5]:


# Display a summary of each dataset
print("Training Dataset:")
print(data_train_df.head())
print("Validation Dataset:")
print(data_validate_df.head())
print("Testing Dataset:")
print(data_test_df.head())


# ### 3. DATA PREPROCESSING

# The dataset has already undergone the primary preprocessing and transformation in the pervious phase. This section is only to display the data and extract features and dates.

# In[6]:


# Convert Date column to a valid Datetime format
data_train_df["Date"] = pd.to_datetime(data_train_df["Date"])
data_validate_df["Date"] = pd.to_datetime(data_validate_df["Date"])
data_test_df["Date"] = pd.to_datetime(data_test_df["Date"])


# In[7]:


# Extract dates from each dataset
data_train_dates = data_train_df["Date"]
data_validate_dates = data_validate_df["Date"]
data_test_dates = data_test_df["Date"]


# In[8]:


# Extract features
features = ["Open", "High", "Low",	"Close", "Adj Close", "Volume"]
data_train_scaled = data_train_df[features].values
data_validate_scaled = data_validate_df[features].values
data_test_scaled = data_test_df[features].values


# ### 4. CONSTRUCTING DATA STRUCTURE

# Given that we are addressing a time series problem involving multiple predictors, our task involves constructing and reshaping the input data to suit the LSTM model.
# 
# This entails setting a sliding time window (sequence size) that determines the number of past observations used to predict the subsequent value.
# 
# In this experiment, we employ a sequence of previous samples (financial days) of all variables to forecast the opening price on the following day.

# In[9]:


# Define a method to construct the input data X and Y
def construct_lstm_data(data, sequence_size, target_attr_idx):
    """
    Construct input data (X) and target data (y) for LSTM model from a pandas DataFrame.

    Parameters:
    -----------
    data : numpy.ndarray
        Input data array of shape (n_samples, n_features).
    
    sequence_size : int
        Number of previous time steps to use as input features for predicting the next time step.
    
    target_attr_idx : int
        Index of column in `data` DataFrame that corresponds to target attribute that LSTM model will predict.

    Returns:
    --------
    data_X : numpy.ndarray
        Array of LSTM input sequences of shape (n_samples - sequence_size, sequence_size, n_features).

    data_y : numpy.ndarray
        Corresponding target values for each input sequence of shape (n_samples - sequence_size,).
    """
    
    # Initialize constructed data variables
    data_X = []
    data_y = []
    
    # Iterate over the dataset
    for i in range(sequence_size, len(data)):
        data_X.append(data[i-sequence_size:i,0:data.shape[1]])
        data_y.append(data[i,target_attr_idx])
        
    # Return constructed variables
    return np.array(data_X), np.array(data_y)


# The construction function is currently available for constructing subsets for training, validation, and testing date that is aligned with the LSTM model input.

# In[10]:


# Define the sequence size
sequence_size = 60

# Construct training data
X_train, y_train = construct_lstm_data(data_train_scaled, sequence_size, 0)


# Because creating input data requires observations from previous samples, constructing subsets for the validation and testing periods suggests we append data from previous periods.
# 
# For instance, to predict the initial stock price in the validation period, we must combine recent actual stock prices (sequence) from the training period. This step is crucial for providing the LSTM model with the expected sequence.
# 
# Same principle applies to stock performance during the testing phase but in this case (and depending on the sequence size), we may need to combine samples from both validation and testing subsets.
# 
# To facilitate this step, we combine the whole scaled dataset together and then select corresponding chunks before constructing the input data sets.

# In[11]:


# Combine scaled datasets all together
data_all_scaled = np.concatenate([data_train_scaled, data_validate_scaled, data_test_scaled], axis=0)

# Calculate data size
train_size = len(data_train_scaled)
validate_size = len(data_validate_scaled)
test_size = len(data_test_scaled)

# Construct validation dataset
X_validate, y_validate = construct_lstm_data(data_all_scaled[train_size-sequence_size:train_size+validate_size,:], sequence_size, 0)

# Construct testing dataset
X_test, y_test = construct_lstm_data(data_all_scaled[-(test_size+sequence_size):,:], sequence_size, 0)


# In[12]:


# Check original data and data splits shapes
print(f"Full Scaled Data: {data_all_scaled.shape}")
print(f"\n Data Train Scaled: {data_train_scaled.shape}")
print(f"> Data Train X: {X_train.shape}")
print(f"> Data Train y: {y_train.shape}")

print(f"\n Data Validate Scaled: {data_validate_scaled.shape}")
print(f"> Data Validate X: {X_validate.shape}")
print(f"> Data Validate y: {y_validate.shape}")

print(f"\n Data Test Scaled: {data_test_scaled.shape}")
print(f"> Data Test X: {X_test.shape}")
print(f"> Data Test y: {y_test.shape}")


# ### 5. TRAINING LSTM MODEL

# #### Building LSTM Model

# In[13]:


# Initializing the model
regressor = Sequential()


# In[14]:


# Add input layer
regressor.add(Input(shape=(X_train.shape[1], X_train.shape[2])))


# In[15]:


# Add first LSTM layer and dropout regularization layer
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(rate = 0.2))


# In[16]:


# Add second LSTM layer and dropout regularization layer
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(rate = 0.2))


# In[17]:


# Add third LSTM layer and dropout regularization layer
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(rate = 0.2))


# In[18]:


# Add forth LSTM layer and dropout regularization layer
regressor.add(LSTM(units = 100))
regressor.add(Dropout(rate = 0.2))


# In[19]:


# Add last dense layer/output layer
regressor.add(Dense(units = 1))


# In[20]:


# Compiling the model
regressor.compile(optimizer = "adam", loss="mean_squared_error")


# #### Training Model

# In[21]:


# Create a checkpoint to monitor the validation loss and save the model with the best performance.
model_location = "..//models//"
model_name = "2025_google_stock_price_lstm.model.keras"
best_model_checkpoint_callback = ModelCheckpoint(
    model_location + model_name, 
    monitor="val_loss", 
    save_best_only=True, 
    mode="min", 
    verbose=0)


# In[22]:


# Training the model
history = regressor.fit(
    x = X_train, 
    y = y_train, 
    validation_data=(X_validate, y_validate), 
    epochs=200, 
    batch_size = 64, 
    callbacks = [best_model_checkpoint_callback])


# #### Performance Evaluation

# In[23]:


# Visualizing model performance during training
plt.figure(figsize=(18, 6))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("LSTM Model Performance")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# The chart above highlights the following observations:
# - Training loss values decrease over the epochs, starting from a higher value and gradually decreasing. This indicates that the model is learning to fit the training data better as training progresses.
# - Similarly, the validation loss values also decrease initially, indicating improvement in the model's ability to generalize to unseen data (validation set).
# - Some fluctuations in the validation loss have been observed, which is quite justifiable due to the high variations and complexity in the price performance over time.
# - There are no signs of overfitting since both loss values are continuously decreasing along with each other over the epochs.
# - Validation loss has maintained lower values than training loss across almost the whole training period, which is quite normal due to the complexity/size of training data.

# ### 6. MODEL INFERENCE

# The LSTM model has been trained and evaluated, making it ready to forecast future price trends.
# 
# As mentioned earlier, we will employ our model to predict Google stock prices during the testing period. To thoroughly assess the model's performance, we will forecast stock performance for the entire period, including testing samples, and compare these predictions with actual prices.

# #### Load Best Model

# In[24]:


# Prepare model location and name
model_location = "..//models//"
model_name = "2025_google_stock_price_lstm.model.keras"

# Load the best performing model
best_model = load_model(model_location + model_name)


# #### Model Prediction

# In[25]:


# Predict stock price for all data splits
y_train_predict = best_model.predict(X_train)
y_validate_predict = best_model.predict(X_validate)
y_test_predict = best_model.predict(X_test)


# #### Inverse Scaling

# One important thing to consider is the scale of the datasets, because the model was trained on scaled data that needs to be transformed back to its original price range.
# 
# Therefore, we have to load the same scaler used for transforming the original dataset and apply it to restore the prediction outputs to their actual distribution.

# In[26]:


# Prepare scaler model name and location
scaler_model_location = "..//models//"
scaler_model_name = "2025_google_stock_price_scaler"
scaler_model_ext = "gz"

# Store the scaler model
sc = joblib.load(scaler_model_location + scaler_model_name + "." + scaler_model_ext)


# The scaler was initially fitted on data with 6 columns encompassing all the features considered. However, during the inverse transformation process, our focus is solely on transforming the "Open" price.
# 
# To circumvent potential errors triggered by the inverse_transform API, we create a container structure of the necessary shape. We populate the first column with predictions and subsequently disregard the others.

# In[27]:


# Restore actual distribution for predicted prices
y_train_inv = sc.inverse_transform(np.concatenate((y_train.reshape(-1,1), np.ones((len(y_train.reshape(-1,1)), 5))), axis=1))[:,0]
y_validate_inv = sc.inverse_transform(np.concatenate((y_validate.reshape(-1,1), np.ones((len(y_validate.reshape(-1,1)), 5))), axis=1))[:,0]
y_test_inv = sc.inverse_transform(np.concatenate((y_test.reshape(-1,1), np.ones((len(y_test.reshape(-1,1)), 5))), axis=1))[:,0]

y_train_predict_inv = sc.inverse_transform(np.concatenate((y_train_predict, np.ones((len(y_train_predict), 5))), axis=1))[:,0]
y_validate_predict_inv = sc.inverse_transform(np.concatenate((y_validate_predict, np.ones((len(y_validate_predict), 5))), axis=1))[:,0]
y_test_predict_inv = sc.inverse_transform(np.concatenate((y_test_predict, np.ones((len(y_test_predict), 5))), axis=1))[:,0]


# #### Display Predictions

# In[28]:


# Define chart colors
train_actual_color = "cornflowerblue"
validate_actual_color = "orange"
test_actual_color = "green"
train_predicted_color = "lightblue"
validate_predicted_color = "peru"
test_predicted_color = "limegreen"


# In[29]:


# Plot actual and predicted price
plt.figure(figsize=(18,6))
plt.plot(data_train_dates[sequence_size:,], y_train_inv, label="Training Data", color=train_actual_color)
plt.plot(data_train_dates[sequence_size:,], y_train_predict_inv, label="Training Predictions", linewidth=1, color=train_predicted_color)

plt.plot(data_validate_dates, y_validate_inv, label="Validation Data", color=validate_actual_color)
plt.plot(data_validate_dates, y_validate_predict_inv, label="Validation Predictions", linewidth=1, color=validate_predicted_color)

plt.plot(data_test_dates, y_test_inv, label="Testing Data", color=test_actual_color)
plt.plot(data_test_dates, y_test_predict_inv, label="Testing Predictions", linewidth=1, color=test_predicted_color)

plt.title("Google Stock Price Predictions With LSTM")
plt.xlabel("Time")
plt.ylabel("Stock Price (USD)")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
plt.legend()
plt.grid(color="lightgray")


# Based on this expanded overview, the model appears to perform very well throughout the observed period, with the predicted trend tends to align closely with actual stock performance across the three training, validation, and testing periods.
# 
# To better assess the model's predictions during the validation and testing periods, let's inspect a zoomed-in version of this chart covering the most recent samples.

# In[30]:


recent_samples = 50
plt.figure(figsize=(18,6))
plt.plot(data_train_dates[-recent_samples:,], y_train_inv[-recent_samples:,], label="Training Data", color=train_actual_color, linewidth=4)
plt.plot(data_train_dates[-recent_samples:,], y_train_predict_inv[-recent_samples:,], label="Training Predictions", linewidth=2, color=train_predicted_color)

plt.plot(data_validate_dates, y_validate_inv, label="Validation Data", color=validate_actual_color, linewidth=4)
plt.plot(data_validate_dates, y_validate_predict_inv, label="Validation Predictions", linewidth=2, color=validate_predicted_color)

plt.plot(data_test_dates, y_test_inv, label="Testing Data", color=test_actual_color, linewidth=4)
plt.plot(data_test_dates, y_test_predict_inv, label="Testing Predictions", linewidth=2, color=test_predicted_color)

plt.title("Google Stock Price Predictions With LSTM (last 50 financial days)")
plt.xlabel("Time")
plt.ylabel("Stock Price (USD)")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
# plt.xticks(rotation=45)
plt.legend()
plt.grid(color="lightgray")


# This closer look underscores the great performance of our LSTM model in predicting Google's stock prices during both validation and testing periods. Indicating the ability of this model to generalize on unseen data that was not included in the training set.
# 
# Despite the exact value predictions might be slightly different from real prices, the strong performance of this model is primarily derived from its ability to consistently mirror the actual trends across almost all phases. This is the most significant takeaway of stock price predictions analysis.

# In[36]:


# Predict price in the future
future_days = 100
last_date = data_test_dates.iloc[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='B')

# Get the most recent sequence of actual data for initial prediction
last_sequence = X_test[-1:].copy()  # Shape: (1, sequence_size, n_features)

# Initialize arrays to store predictions
future_predictions = []

# Calculate average daily changes from recent history
recent_data = data_test_scaled[-30:]  # Last 30 days
avg_daily_changes = np.mean(np.abs(recent_data[1:] - recent_data[:-1]), axis=0)

# Predict iteratively for future days
for _ in range(future_days):
    # Make prediction
    next_pred = best_model.predict(last_sequence, verbose=0)
    future_predictions.append(next_pred[0,0])
    
    # Create new data point based on prediction
    new_point = last_sequence[0,-1,:].copy()
    new_point[0] = next_pred[0,0]  # Open
    
    # Update other features using historical relationships
    new_point[1] = new_point[0] + avg_daily_changes[1]  # High
    new_point[2] = new_point[0] - avg_daily_changes[2]  # Low
    new_point[3] = new_point[0] + avg_daily_changes[3]  # Close
    new_point[4] = new_point[3]  # Adj Close
    new_point[5] = np.mean(last_sequence[0,:,5])  # Volume (average of last sequence)
    
    # Update the sequence by rolling and adding new point
    last_sequence = np.roll(last_sequence, -1, axis=1)
    last_sequence[0,-1,:] = new_point

# Convert predictions to original scale
future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions_padded = np.concatenate((future_predictions, np.ones((len(future_predictions), 5))), axis=1)
future_predictions_inv = sc.inverse_transform(future_predictions_padded)[:,0]

# Plot results
plt.figure(figsize=(18,6))
plt.plot(data_test_dates[-30:], y_test_inv[-30:], label="Historical Data", color=test_actual_color, linewidth=2)
plt.plot(data_test_dates[-30:], y_test_predict_inv[-30:], label="Historical Predictions", color=test_predicted_color, linewidth=2)
plt.plot(future_dates, future_predictions_inv, label="Future Predictions", color='purple', linestyle='--', linewidth=2)

plt.title("Google Stock Price Future Predictions With LSTM")
plt.xlabel("Time")
plt.ylabel("Stock Price (USD)")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()

