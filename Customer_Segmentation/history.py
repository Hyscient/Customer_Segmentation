# -*- coding: utf-8 -*-
# *** Spyder Python Console History Log ***

## ---(Mon Apr 24 10:22:17 2023)---
runfile('C:/Users/37066/untitled0.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/untitled0.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/untitled0.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/untitled0.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/untitled1.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/untitled0.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/untitled0.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/untitled0.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/untitled0.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/untitled2.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/untitled2.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/untitled2.py', wdir='C:/Users/37066')
runcell(0, 'C:/Users/37066/untitled2.py')
runfile('C:/Users/37066/untitled2.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/untitled3.py', wdir='C:/Users/37066')

plt.figure(figsize=(12,6))
plt.scatter(df_year['value'], df_year['population'])
plt.title('CO2 Emissions vs. Population (' + str(year) + ')')
plt.xlabel('CO2 Emissions (metric tons per capita)')
plt.ylabel('Population')
plt.show()
runfile('C:/Users/37066/untitled3.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/untitled3.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/untitled3.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/untitled4.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/untitled4.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/CO2 Analytics.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/CO2 Analytics.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/CO2 Analytics.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/CO2 Analytics.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/CO2 Analytics.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/CO2 Analytics.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/CO2 Analytics.py', wdir='C:/Users/37066')
pd.concat
runfile('C:/Users/37066/CO2 Analytics.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/CO2 Analytics.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/CO2 Analytics.py', wdir='C:/Users/37066')

## ---(Tue Apr 25 16:36:49 2023)---
runfile('C:/Users/37066/untitled0.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/untitled0.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/untitled0.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/untitled0.py', wdir='C:/Users/37066')
runfile('C:/Users/37066/untitled0.py', wdir='C:/Users/37066')

## ---(Tue Apr 25 18:00:12 2023)---
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')

## ---(Fri Apr 28 12:30:29 2023)---
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runcell(0, 'C:/Users/37066/.spyder-py3/temp.py')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runcell(0, 'C:/Users/37066/.spyder-py3/temp.py')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
print(df.columns)

cohort_data = df.groupby(['CohortMonth', 'MonthOffset'])

# Calculate the number of unique active users in each group
cohort_counts = cohort_data['user_id'].nunique()

# Calculate the retention rates for each cohort group
cohort_retention = cohort_counts.unstack().divide(cohort_counts.groupby(level=0).first(), axis=0)
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')

print(df.columns)

cohort_data = df.groupby(['CohortMonth', 'MonthOffset'])

# Calculate the number of unique active users in each group
cohort_counts = cohort_data['user_id'].nunique()

# Calculate the retention rates for each cohort group
cohort_retention = cohort_counts.unstack().divide(cohort_counts.groupby(level=0).first(), axis=0)
df['CohortMonth'] = df.groupby('user_id')['event_time'].transform('min').dt.to_period('M')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
print(retention.columns)
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
october_retention = cohort_data[cohort_data['CohortMonth'] == '2019-10']
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')

plt.figure(figsize=(10, 8))
plt.title('Cohorts: User Retention')
sns.heatmap(data=retention,
            annot=True,
            fmt='.0%',
            vmin=0.0,
            vmax=0.5,
            cmap='BuGn')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Load the data from CSV files, limiting to the first 10000 rows
df1 = pd.read_csv('2019-Oct.csv', nrows=10000)
df2 = pd.read_csv('2019-Nov.csv', nrows=10000)

# Concatenate the two dataframes
df = pd.concat([df1, df2], ignore_index=True)

# Convert event_time column to datetime
df['event_time'] = pd.to_datetime(df['event_time'])
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')

df_cohort['CLV'] = df_cohort['avg_purchase_value'] * df_cohort['avg_purchases_per_customer_per_year'] * df_cohort['avg_customer_lifespan']

plt.figure(figsize=(10, 8))
plt.title('Cohorts: User Retention')
sns.heatmap(data=retention,
            annot=True,
            fmt='.0%',
            vmin=0,
            vmax=0.5,
            cmap='BuGn')

# Add labels to the x-axis and y-axis
plt.xlabel('Months Since Initial Purchase')
plt.ylabel('Cohort Month')

plt.show()
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')

df1 = pd.read_csv('2019-Oct.csv')
df2 = pd.read_csv('2019-Nov.csv')
df = pd.concat([df1, df2], ignore_index=True)

# Convert event_time to datetime
df['event_time'] = pd.to_datetime(df['event_time'])

# Extract year-month from event_time
df['year_month'] = df['event_time'].dt.strftime('%Y-%m')

# Create a new column for total revenue per order
df['revenue'] = df['price'] * df['quantity']

# Group by user_id and year_month to get monthly revenue per user
grouped = df.groupby(['user_id', 'year_month']).agg({'revenue': 'sum'}).reset_index()

# Calculate customer lifetime value (CLV) for each user
grouped['total_revenue'] = grouped.groupby('user_id')['revenue'].transform('sum')
grouped['order_count'] = grouped.groupby('user_id')['revenue'].transform('count')
grouped['avg_order_value'] = grouped['total_revenue'] / grouped['order_count']
grouped['customer_lifetime_value'] = grouped['avg_order_value'] * 12

# Create cohorts based on the user's first purchase month
grouped['cohort_month'] = grouped.groupby('user_id')['year_month'].transform('min')
grouped['cohort_lifetime'] = (grouped['year_month'].astype('datetime64[ns]').dt.to_period('M') 
                              - grouped['cohort_month'].astype('datetime64[ns]').dt.to_period('M')).apply(lambda x: x.n)

# Group by cohort_month and cohort_lifetime to get the number of unique customers and total revenue per cohort
cohorts = grouped.groupby(['cohort_month', 'cohort_lifetime']).agg({
    'user_id': 'nunique',
    'revenue': 'sum'
}).reset_index()

# Calculate average order value (AOV) for each cohort
cohorts['average_order_value'] = cohorts['revenue'] / cohorts['user_id']

# Create a pivot table to visualize the cohorts
pivot = cohorts.pivot_table(index='cohort_month', columns='cohort_lifetime', values='average_order_value', 
                            aggfunc='mean')

# Plot the pivot table using a heatmap
sns.set(style='white')
plt.figure(figsize=(12, 8))
plt.title('Cohorts: Average Order Value')
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='Blues')
plt.show()
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
print(df.columns)
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')

df['quantity'] = 1
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
print(df.columns)
import pandas as pd
import numpy as np

# Read the data
df1 = pd.read_csv('2019-Oct.csv', nrows=10000)
df2 = pd.read_csv('2019-Nov.csv', nrows=10000)

# Concatenate the data
df = pd.concat([df1, df2])

# Convert event_time to datetime format
df['event_time'] = pd.to_datetime(df['event_time'])

# Add a new column "month_year" to the dataframe
df['month_year'] = df['event_time'].dt.to_period('M')

# Add a new column "quantity"
df['quantity'] = 1

print(df.head())
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
import pandas as pd
import numpy as np

# Read the data
df1 = pd.read_csv('2019-Oct.csv', nrows=10000)
df2 = pd.read_csv('2019-Nov.csv', nrows=10000)

# Concatenate the data
df = pd.concat([df1, df2])

# Convert event_time to datetime format
df['event_time'] = pd.to_datetime(df['event_time'])

# Add a new column "month_year" to the dataframe
df['month_year'] = df['event_time'].dt.to_period('M')

# Add a new column "quantity"
df['quantity'] = 1

print(df.head())
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')

df1 = pd.read_csv('2019-Oct.csv', nrows=10000)
df2 = pd.read_csv('2019-Nov.csv', nrows=10000)

# Concatenate the two dataframes
df = pd.concat([df1, df2], ignore_index=True)

# Convert event_time column to datetime
df['event_time'] = pd.to_datetime(df['event_time'])

# Add a TotalSpent column based on the transaction amount
df['TotalSpent'] = df['price'] * df['quantity']

# Calculate the Recency, Frequency, and MonetaryValue for each user
snapshot_date = df['event_time'].max() + dt.timedelta(days=1)
rfm = df.groupby('user_id').agg({
    'event_time': lambda x: (snapshot_date - x.max()).days,
    'order_id': 'nunique',
    'TotalSpent': 'sum'
})
rfm.rename(columns={
    'event_time': 'Recency',
    'order_id': 'Frequency',
    'TotalSpent': 'MonetaryValue'
}, inplace=True)

# Apply quantiles to each RFM metric
quantiles = rfm.quantile(q=[0.25, 0.5, 0.75])
quantiles = quantiles.to_dict()

def r_score(x, c):
    if x <= c[0.25]:
        return 4
    elif x <= c[0.5]:
        return 3
    elif x <= c[0.75]: 
        return 2
    else:
        return 1

def fm_score(x, c):
    if x <= c[0.25]:
        return 1
    elif x <= c[0.5]:
        return 2
    elif x <= c[0.75]: 
        return 3
    else:
        return 4

rfm['R'] = rfm['Recency'].apply(r_score, args=(quantiles['Recency'],))
rfm['F'] = rfm['Frequency'].apply(fm_score, args=(quantiles['Frequency'],))
rfm['M'] = rfm['MonetaryValue'].apply(fm_score, args=(quantiles['MonetaryValue'],))

# Combine the RFM scores into a single string
rfm['RFM'] = rfm['R'].map(str) + rfm['F'].map(str) + rfm['M'].map(str)

# Print the top 10 RFM segments
print(rfm.groupby('RFM').size().sort_values(ascending=False).head(10))
import pandas as pd
# Load the datasets
df1 = pd.read_csv('2019-Oct.csv', nrows=10000)
df2 = pd.read_csv('2019-Nov.csv', nrows=10000)

# Concatenate the datasets
df = pd.concat([df_oct, df_nov], ignore_index=True)

# Convert event_time column to datetime
df['event_time'] = pd.to_datetime(df['event_time'])

# Calculate recency
recency = df.groupby('user_id')['event_time'].max().reset_index()
recency['recency'] = (recency['event_time'].max() - recency['event_time']).dt.days
recency = recency[['user_id', 'recency']]

# Calculate frequency
frequency = df.groupby('user_id')['event_type'].count().reset_index()
frequency.columns = ['user_id', 'frequency']

# Calculate monetary value
monetary_value = df.groupby('user_id')['price'].sum().reset_index()
monetary_value.columns = ['user_id', 'monetary_value']

# Merge the recency, frequency, and monetary_value dataframes
rfm = recency.merge(frequency, on='user_id').merge(monetary_value, on='user_id')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')

df['event_time'] = pd.to_datetime(df['event_time'])

# Calculate recency
recency = df.groupby('user_id')['event_time'].max().reset_index()
recency['recency'] = (recency['event_time'].max() - recency['event_time']).dt.days
recency = recency[['user_id', 'recency']]

# Calculate frequency
frequency = df.groupby('user_id')['event_type'].count().reset_index()
frequency.columns = ['user_id', 'frequency']

# Calculate monetary value
monetary_value = df.groupby('user_id')['price'].sum().reset_index()
monetary_value.columns = ['user_id', 'monetary_value']

# Merge the recency, frequency, and monetary_value dataframes
rfm = recency.merge(frequency, on='user_id').merge(monetary_value, on='user_id')
import pandas as pd

df_oct = pd.read_csv('2019-Oct.csv')
df_nov = pd.read_csv('2019-Nov.csv')

df = pd.concat([df_oct, df_nov], ignore_index=True)

# Convert event_time column to datetime
df['event_time'] = pd.to_datetime(df['event_time'])

# Calculate recency
recency = df.groupby('user_id')['event_time'].max().reset_index()
recency['recency'] = (recency['event_time'].max() - recency['event_time']).dt.days
recency = recency[['user_id', 'recency']]

# Calculate frequency
frequency = df.groupby('user_id')['event_type'].count().reset_index()
frequency.columns = ['user_id', 'frequency']

# Calculate monetary value
monetary_value = df.groupby('user_id')['price'].sum().reset_index()
monetary_value.columns = ['user_id', 'monetary_value']

# Merge the recency, frequency, and monetary_value dataframes
rfm = recency.merge(frequency, on='user_id').merge(monetary_value, on='user_id')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')

r_labels = range(4, 0, -1)
r_quartiles = pd.qcut(rfm['recency'], q=4, labels=r_labels)

f_labels = range(1, 5)
f_quartiles = pd.qcut(rfm['frequency'], q=4, labels=f_labels)

m_labels = range(1, 5)
m_quartiles = pd.qcut(rfm['monetary_value'], q=4, labels=m_labels)

# Add the quartiles to the dataframe as new variables
rfm = rfm.assign(R=r_quartiles, F=f_quartiles, M=m_quartiles)

# Concatenate the quartiles to create the RFM segment
rfm['RFM_segment'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)

# Calculate the RFM score
rfm['RFM_score'] = rfm[['R', 'F', 'M']].sum(axis=1)

data = pd.DataFrame({'value': np.random.randint(0, 100, 100)})

# Define bin edges for the value column
bins = [-1, 25, 50, 75, 100]

# Use pandas cut function to bin the values
data['bin'] = pd.cut(data['value'], bins=bins, duplicates='drop')

# Print the counts for each bin
print(data['bin'].value_counts())
df1 = pd.read_csv('2019-Oct.csv', nrows=10000)
df2 = pd.read_csv('2019-Nov.csv', nrows=10000)


# Concatenate the two dataframes
df = pd.concat([df1, df2], ignore_index=True)

# Convert event_time column to datetime
df['event_time'] = pd.to_datetime(df['event_time'])

# Calculate recency
recency = df.groupby('user_id')['event_time'].max().reset_index()
recency['recency'] = (recency['event_time'].max() - recency['event_time']).dt.days
recency = recency[['user_id', 'recency']]

# Calculate frequency
frequency = df.groupby('user_id')['event_type'].count().reset_index()
frequency.columns = ['user_id', 'frequency']

# Calculate monetary value
monetary_value = df.groupby('user_id')['price'].sum().reset_index()
monetary_value.columns = ['user_id', 'monetary_value']

# Merge the recency, frequency, and monetary_value dataframes
rfm = recency.merge(frequency, on='user_id').merge(monetary_value, on='user_id')
runfile('C:/Users/37066/.spyder-py3/cohort analysis.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')

r_bins = [-1, 30, 60, 90, rfm['recency'].max()+1]
f_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, rfm['frequency'].max()+1]
m_bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, rfm['monetary_value'].max()+1]

# Use pandas cut function to bin the RFM values
rfm['R'] = pd.cut(rfm['recency'], bins=r_bins, labels=[4, 3, 2, 1])
rfm['F'] = pd.cut(rfm['frequency'], bins=f_bins, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
rfm['M'] = pd.cut(rfm['monetary_value'], bins=m_bins, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Concatenate the R, F, and M values into a single column
rfm['RFM'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)

# Define the segments based on the RFM values
segments = {
    'Best Customers': ['111', '211', '311', '411', '511', '611', '711', '811', '911', '1011'],
    'Loyal Customers': ['141', '142', '143', '241', '242', '243', '341', '342', '343', '441', '442', '443'],
    'Big Spenders': ['114', '124', '134', '144', '154', '164', '174', '184', '194', '1044'],
    'Almost Lost': ['411', '412', '413', '421', '422', '423', '431', '432', '433'],
    'Lost Customers': ['311', '312', '313', '321', '322', '323', '331', '332', '333'],
    'Lost Cheap Customers': ['111', '112', '113', '121', '122', '123', '131', '132', '133', '141', '142', '143']
}

# Assign each customer to a segment
rfm['Segment'] = np.nan
for segment, rfm_values in segments.items():
    rfm.loc[rfm['RFM'].isin(rfm_values), 'Segment'] = segment
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')

def assign_segments(row):
    for segment, rfm_codes in segments.items():
        if row['RFM'] in rfm_codes:
            return segment
    return 'Other'

# Apply the function to each row of the rfm dataframe
rfm['segment'] = rfm.apply(assign_segments, axis=1)
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
segment_counts = rfm['Segment'].value_counts()
plt.bar(segment_counts.index, segment_counts.values)
plt.title('Distribution of Customers by Segment')
plt.xlabel('Segment')
plt.ylabel('Number of Customers')
plt.show()
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled1.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled1.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled1.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled1.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/customer_segmentation.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/churn_analysis.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/churn_analysis.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/churn_analysis.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/churn_analysis.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/churn_analysis.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/churn_analysis.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/churn_analysis.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/churn_analysis.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/rfm analysis.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled3.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled3.py', wdir='C:/Users/37066/.spyder-py3')
pip install fbprophet
runfile('C:/Users/37066/.spyder-py3/untitled3.py', wdir='C:/Users/37066/.spyder-py3')
python -m pip install prophet
runfile('C:/Users/37066/.spyder-py3/untitled3.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled3.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled3.py', wdir='C:/Users/37066/.spyder-py3')
python -m pip install prophet
pip install prophet
runfile('C:/Users/37066/.spyder-py3/untitled3.py', wdir='C:/Users/37066/.spyder-py3')
from prophet import Prophet
runfile('C:/Users/37066/.spyder-py3/untitled3.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled3.py', wdir='C:/Users/37066/.spyder-py3')
print(df.isnull().sum())
df.fillna(value, inplace=True)
df.fillna(0, inplace=True)
print(df.isnull().sum())
runfile('C:/Users/37066/.spyder-py3/untitled3.py', wdir='C:/Users/37066/.spyder-py3')
df.dropna(inplace=True)
df.dropna(subset=['y'], inplace=True)
runfile('C:/Users/37066/.spyder-py3/untitled3.py', wdir='C:/Users/37066/.spyder-py3')
df.dropna(inplace=True)
df.dropna(subset=['column1', 'column2'], inplace=True)
runfile('C:/Users/37066/.spyder-py3/untitled3.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled3.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled3.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled3.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled3.py', wdir='C:/Users/37066/.spyder-py3')
!pip install fbprophet
pip install cython
!pip install fbprophet
pip install cython
!pip install fbprophet
pip install pystan
runfile('C:/Users/37066/.spyder-py3/untitled3.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled3.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled3.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled3.py', wdir='C:/Users/37066/.spyder-py3')
pip install fbprophet
git clone https://github.com/facebook/prophet.git
>git clone https://github.com/facebook/prophet.git
runfile('C:/Users/37066/.spyder-py3/untitled3.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled3.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled3.py', wdir='C:/Users/37066/.spyder-py3')
pip install fbprophet

## ---(Sat Apr 29 16:45:22 2023)---
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/cohort analysis.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled0.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled0.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled0.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled0.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled0.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled0.py', wdir='C:/Users/37066/.spyder-py3')

y_pred = model.predict(X_test)

# calculate R-squared
r_squared = r2_score(y_test, y_pred)

print("R-squared:", r_squared)
runfile('C:/Users/37066/.spyder-py3/untitled1.py', wdir='C:/Users/37066/.spyder-py3')
pip install xgboost
runfile('C:/Users/37066/.spyder-py3/untitled1.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled1.py', wdir='C:/Users/37066/.spyder-py3')

filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
runfile('C:/Users/37066/.spyder-py3/prediction_training.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/prediction_training.py', wdir='C:/Users/37066/.spyder-py3')

filename = 'xgb_model.sav'
pickle.dump(model, open(filename, 'wb'))

# Load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
runfile('C:/Users/37066/.spyder-py3/untitled2.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled2.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled2.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled2.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled2.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled2.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled2.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled2.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled2.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled2.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled2.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled1.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/prediction_training.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/cohort analysis.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/rfm_analysis.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/churn_analysis.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/customer_segmentation.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/customer_segmentation.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/cohort analysis.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/predictive_analysis.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/rfm_analysis.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/predictive_analysis.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/temp.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/predictive_analysis.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/predictive_analysis.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/prediction_training.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/prediction_training.py', wdir='C:/Users/37066/.spyder-py3')
runfile('C:/Users/37066/.spyder-py3/untitled8.py', wdir='C:/Users/37066/.spyder-py3')