import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV files, limiting to the first 10000 rows
df1 = pd.read_csv('2019-Oct.csv', nrows=10000)
df2 = pd.read_csv('2019-Nov.csv', nrows=10000)

# Concatenate the two dataframes
data = pd.concat([df1, df2], ignore_index=True)

# Convert 'event_time' column to datetime format
data['event_time'] = pd.to_datetime(data['event_time'])

# Access year and month of the datetime column
data['year_month'] = data['event_time'].dt.strftime('%Y-%m')

# Filter out rows that contain non-numeric values in the 'price' column
data = data[pd.to_numeric(data['price'], errors='coerce').notnull()]

# Convert the 'price' column to float
data['price'] = data['price'].astype(float)

# Get the number of unique users who made a purchase each month
monthly_active_users = data.groupby('year_month')['user_session'].nunique()

# Get the number of unique users who made a purchase in the previous month
prev_month_active_users = monthly_active_users.shift(1)

# Compute the monthly churn rate
monthly_churn_rate = (prev_month_active_users - monthly_active_users) / prev_month_active_users * 100
monthly_churn_rate.fillna(0, inplace=True)

# Plot the monthly churn rate
plt.figure(figsize=(8, 6))
plt.plot(monthly_churn_rate.index, monthly_churn_rate.values)
plt.ylabel('Churn Rate (%)')
plt.title('Monthly Churn Rate')
plt.show()

monthly_active_users = data.groupby(data['event_time'].dt.strftime('%Y-%m'))['user_id'].nunique()
plt.figure(figsize=(8,6))
plt.plot(monthly_active_users)
plt.xticks(rotation=45)
plt.xlabel('Month')
plt.ylabel('Monthly Active Users')
plt.title('Monthly Active Users')
plt.show()

clv_data = data.groupby('user_id')['price'].sum()
plt.figure(figsize=(8,6))
plt.hist(clv_data, bins=50)
plt.xlabel('Customer Lifetime Value (CLV)')
plt.ylabel('Number of Customers')
plt.title('Distribution of Customer Lifetime Value (CLV)')
plt.show()


# Set parameters
num_customers = 3000
avg_clv = 20000
churn_rate = 0.05
time_horizon = 5 # years

# Calculate expected revenue
revenue_per_customer = avg_clv / time_horizon
expected_revenue = num_customers * revenue_per_customer

# Calculate revenue lost due to churn
revenue_lost = expected_revenue * churn_rate

# Calculate revenue retained
revenue_retained = expected_revenue - revenue_lost

# Create a pie chart
labels = ['Revenue Retained', 'Revenue Lost to Churn']
sizes = [revenue_retained, revenue_lost]
colors = ['#66b3ff', '#ff6666']
explode = (0.1, 0)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')

plt.title('Expected Revenue\n${:,.0f} Total Revenue\n${:,.0f} Lost to Churn\n${:,.0f} Retained Revenue'.format(expected_revenue, revenue_lost, revenue_retained))
plt.show()
