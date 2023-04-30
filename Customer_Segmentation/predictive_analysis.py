import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the data
data_oct = pd.read_csv('2019-Oct.csv', nrows=10000)
data_nov = pd.read_csv('2019-Oct.csv', nrows=10000)
data = pd.concat([data_oct, data_nov])

# Filter the data to only include purchases
purchases = data[data['event_type'] == 'purchase']

# Aggregate the data by user_id
user_data = purchases.groupby('user_id').agg({'price': ['sum', 'mean'], 'event_time': 'count'})

# Rename the columns
user_data.columns = ['total_spent', 'average_spent', 'purchase_count']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(user_data.drop('total_spent', axis=1), user_data['total_spent'], test_size=0.2)

# Train an XGBoost model
model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate R-squared
r_squared = r2_score(y_test, y_pred)

print("R-squared:", r_squared)


