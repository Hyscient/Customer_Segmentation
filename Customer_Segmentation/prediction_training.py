import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pickle

# Load the data
data_oct = pd.read_csv('2019-Oct.csv', nrows=10000)
data_nov = pd.read_csv('2019-Oct.csv', nrows=10000)

# Concatenate the two datasets
data = pd.concat([data_oct, data_nov])

# Drop unnecessary columns
data = data.drop(['event_time', 'event_type', 'product_id', 'category_id', 'category_code', 'user_session'], axis=1)

# Convert the brand column to numerical values
brand_dict = {k:i for i, k in enumerate(data['brand'].unique())}
data['brand'] = data['brand'].apply(lambda x: brand_dict[x])

# Convert the user_id column to numerical values
user_dict = {k:i for i, k in enumerate(data['user_id'].unique())}
data['user_id'] = data['user_id'].apply(lambda x: user_dict[x])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop(['price'], axis=1), data['price'], test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print('R-squared:', r2_score(y_test, y_pred))



plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()


# Save the model to disk
filename = 'xgb_model.sav'
pickle.dump(model, open(filename, 'wb'))

# Load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))



