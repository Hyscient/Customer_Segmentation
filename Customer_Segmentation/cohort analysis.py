import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import datetime as dt

# Load the data from CSV files, limiting to the first 10000 rows
df1 = pd.read_csv('2019-Oct.csv', nrows=10000)
df2 = pd.read_csv('2019-Nov.csv', nrows=10000)

# Concatenate the two dataframes
df = pd.concat([df1, df2], ignore_index=True)

# Convert event_time column to datetime
df['event_time'] = pd.to_datetime(df['event_time'])

# Add a CohortMonth column based on the user's first transaction month
df['CohortMonth'] = df.groupby('user_id')['event_time'].transform('min').dt.to_period('M')

# Add a TransactionMonth column based on the transaction date
df['TransactionMonth'] = df['event_time'].dt.to_period('M')

# Calculate the time offset in months between the user's first transaction and each subsequent transaction
def get_month_offset(df):
    cohort_month = df['CohortMonth'].min()
    transaction_month = df['TransactionMonth']
    return (transaction_month - cohort_month).apply(lambda x: x.n).astype(int)

df['MonthOffset'] = df.groupby('user_id').apply(get_month_offset).reset_index(drop=True)

# Print the first few rows to verify the data
print(df.head())

# Drop the columns if they already exist
df = df.drop(['CohortMonth', 'TransactionMonth'], axis=1)

# Add the CohortMonth column
df['CohortMonth'] = df.groupby('user_id')['event_time'].transform('min').dt.to_period('M')

# Add the TransactionMonth column
df['TransactionMonth'] = df['event_time'].dt.to_period('M')

# Calculate the MonthOffset column
df['MonthOffset'] = (df['TransactionMonth'] - df['CohortMonth']).apply(lambda x: x.n)

# Print the first 5 rows of the updated DataFrame
print(df.head())

# Group the data by CohortMonth, MonthOffset, and event_type
cohort_data = df.groupby(['CohortMonth', 'MonthOffset', 'event_type'])['user_id'].nunique().reset_index()

# Create a pivot table with the number of unique users for each cohort group
cohort_counts = cohort_data.pivot_table(index='CohortMonth', columns='MonthOffset', values='user_id')

# Print the cohort counts
print(cohort_counts)


warnings.filterwarnings('ignore')

# Create a pivot table with retention rates
cohort_pivot = pd.pivot_table(
    data=df,
    values='user_id',
    index='CohortMonth',
    columns='MonthOffset',
    aggfunc=pd.Series.nunique,
)

# Divide each column by the first column to get retention rates
cohort_size = cohort_pivot.iloc[:, 0]
retention = cohort_pivot.divide(cohort_size, axis=0)

# Count the number of unique users in the October 2019 cohort
cohort_group_size = df[df['CohortMonth'] == '2019-10']['user_id'].nunique()

# Calculate the number of active users for each month
cohorts = df.groupby(['CohortMonth', 'MonthOffset'])
cohort_data = cohorts['user_id'].apply(pd.Series.nunique).reset_index()
cohort_data.rename(columns={'user_id': 'ActiveUsers'}, inplace=True)

# Calculate the retention rate for the October 2019 cohort
october_retention = cohort_data[cohort_data['CohortMonth'] == '2019-10']
october_retention['Retention'] = october_retention['ActiveUsers'] / cohort_group_size

print(october_retention)

october_retention = cohort_data[cohort_data['CohortMonth'] == '2019-10']

november_retention = cohort_data[cohort_data['CohortMonth'] == '2019-11']
november_retention['Retention'] = november_retention['ActiveUsers'] / cohort_group_size
print(november_retention)


# Plot the retention rates as a heatmap
plt.figure(figsize=(10, 8))
plt.title('Retention Rates')
sns.heatmap(retention, annot=True, fmt='.0%', cmap='YlGnBu')
plt.show()


# Set the plot style
plt.style.use('seaborn-darkgrid')

# Plot the retention rates for each cohort
retention.plot(figsize=(10,5))

# Set the plot title and axis labels
plt.title('Monthly Cohorts: User Retention')
plt.xlabel('Months Since Initial Purchase')
plt.ylabel('Retention Rate')

# Show the plot
plt.show()


# Create the heatmap using seaborn
plt.figure(figsize=(10, 8))
plt.title('Cohorts: User Retention')
sns.heatmap(retention, annot=True, fmt='.0%', cmap='RdYlGn', vmin=0, vmax=0.5)
plt.show()


# Set the figure size
plt.figure(figsize=(10, 8))

# Create the heatmap
sns.heatmap(retention, annot=True, fmt='.0%', cmap='RdYlGn')

# Set the title and axis labels
plt.title('Monthly Cohorts: User Retention')
plt.xlabel('Months Since Initial Purchase')
plt.ylabel('Cohort Month')

# Show the plot
plt.show()


# Create the heatmap
plt.figure(figsize=(10, 8))
plt.title('Cohorts: User Retention')
sns.heatmap(data=retention,
            annot=True,
            fmt='.0%',
            vmin=0,
            vmax=0.5,
            cmap='BuGn')
plt.show()



# Plot the heatmap
sns.set(style='white')
plt.figure(figsize=(12, 8))
plt.title('Retention Rates')
sns.heatmap(retention, annot=True, fmt='.0%', vmin=0, vmax=0.5, cmap='BuGn')
plt.show()



# Set up the figure
plt.figure(figsize=(10, 8))
plt.title('Cohorts: User Retention')
sns.heatmap(data=retention,
            annot=True,
            fmt='.0%',
            vmin=0.0,
            vmax=0.5,
            cmap='BuGn')
plt.show()

# Customize the plot
plt.title('Cohorts: User Retention')
plt.xlabel('MonthOffset')
plt.ylabel('CohortMonth')
plt.show()


# Set the figure size
plt.figure(figsize=(12, 8))

# Add a title
plt.title('Cohorts: User Retention')

# Create the heatmap
sns.heatmap(retention, annot=True, fmt='.0%', vmin=0.0, vmax=0.5, cmap='BuGn')


plt.savefig('cohort_retention.png')

plt.show()

# Create the heatmap
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


# Create a bar chart
x = ['A', 'B', 'C']
y = [1, 2, 3]
plt.bar(x, y)
plt.title('My Bar Chart')

# Save the visualization as a PNG file
plt.savefig('my_bar_chart.png')

