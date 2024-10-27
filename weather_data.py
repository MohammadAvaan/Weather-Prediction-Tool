import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Fetch weather data for Lucknow
API_KEY = '243e5d85b3e51d262c0a4e16f490db53'  # OpenWeatherMap API key
CITY = 'Lucknow'
BASE_URL = "https://api.openweathermap.org/data/2.5/"
url = f"{BASE_URL}weather?q={CITY}&appid={API_KEY}&units=metric"

response = requests.get(url)
data = response.json()  # Get and parse JSON response from the API  

# Check for successful response
if response.status_code == 200:
    main = data['main']
    temperature = main['temp']
    humidity = main['humidity']
    wind_speed = data['wind']['speed']
    weather_description = data['weather'][0]['description']

    print(f"City: {CITY}")
    print(f"Temperature: {temperature}°C")
    print(f"Humidity: {humidity}%")
    print(f"Wind Speed: {wind_speed} m/s")
    print(f"Weather Description: {weather_description.capitalize()}")
else:
    print("Error in the HTTP request. Please check your API key or city name.")

# Load historical data
lucknow_data = pd.read_csv('C:/Users/Admin/Documents/weatherprediction/lucknow_data.csv')
bangalore_data = pd.read_csv('C:/Users/Admin/Documents/weatherprediction/banglore_data.csv')

# Handle missing values
for column in ['tavg', 'tmin', 'tmax', 'prcp']:
    lucknow_data[column] = lucknow_data[column].fillna(lucknow_data[column].mean())
    bangalore_data[column] = bangalore_data[column].fillna(bangalore_data[column].mean())

# Convert 'time' column to datetime
lucknow_data['time'] = pd.to_datetime(lucknow_data['time'], format='%d-%m-%Y')
bangalore_data['time'] = pd.to_datetime(bangalore_data['time'], format='%d-%m-%Y')

# Remove duplicates
lucknow_data.drop_duplicates(inplace=True)
bangalore_data.drop_duplicates(inplace=True)

# Step 1: Exploratory Data Analysis (EDA)

# Visualize Missing Values for Lucknow
plt.figure(figsize=(12, 6))
sns.heatmap(lucknow_data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values in Lucknow Data')
plt.show()

# Distribution of average temperature in Lucknow
plt.figure(figsize=(12, 6))
sns.histplot(lucknow_data['tavg'], bins=30, kde=True)
plt.title('Distribution of Average Temperature in Lucknow')
plt.xlabel('Temperature (°C)')
plt.ylabel('Frequency')
plt.show()

# Time series plot for Lucknow: Avg Temperature
plt.figure(figsize=(12, 6))
plt.plot(lucknow_data['time'], lucknow_data['tavg'], label='Avg Temperature', color='blue')
plt.title('Average Temperature in Lucknow Over Time')
plt.xlabel('Date')
plt.ylabel('Avg Temperature (°C)')
plt.legend()
plt.grid()
plt.show()

# Prepare data for predictive analysis
lucknow_data['time_ordinal'] = lucknow_data['time'].map(pd.Timestamp.toordinal)
X = lucknow_data[['time_ordinal']]
y = lucknow_data['tavg']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Predict future temperature (for the next 10 days)
future_dates = [lucknow_data['time'].iloc[-1] + pd.Timedelta(days=i) for i in range(1, 11)]
future_ordinal = np.array([date.toordinal() for date in future_dates]).reshape(-1, 1)
future_predictions = model.predict(future_ordinal)

# Ensure future_predictions is a DataFrame for consistency
future_predictions_df = pd.DataFrame(future_predictions, columns=['Predicted Temperature'])

# Visualize the predictions
plt.figure(figsize=(12, 6))
plt.plot(lucknow_data['time'], lucknow_data['tavg'], label='Actual Avg Temperature', color='blue')
plt.scatter(lucknow_data['time'].iloc[X_test.index], predictions, color='red', label='Predicted Avg Temperature', marker='x')
plt.title('Actual vs Predicted Average Temperature in Lucknow')
plt.xlabel('Date')
plt.ylabel('Avg Temperature (°C)')
plt.legend()
plt.grid()
plt.show()

# Predict future temperature (for the next 10 days)
future_dates = [lucknow_data['time'].iloc[-1] + pd.Timedelta(days=i) for i in range(1, 11)]
future_ordinal = np.array([date.toordinal() for date in future_dates]).reshape(-1, 1)
future_predictions = model.predict(future_ordinal)

# Visualize future predictions
plt.figure(figsize=(12, 6))
plt.plot(lucknow_data['time'], lucknow_data['tavg'], label='Historical Avg Temperature', color='blue')
plt.plot(future_dates, future_predictions, label='Predicted Avg Temperature (Next 10 Days)', color='orange', linestyle='--')
plt.title('Temperature Prediction for Next 10 Days in Lucknow')
plt.xlabel('Date')
plt.ylabel('Avg Temperature (°C)')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.show()
