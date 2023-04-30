import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV files
df1 = pd.read_csv('2019-Oct.csv', nrows=10000)
df2 = pd.read_csv('2019-Nov.csv', nrows=10000)

# Concatenate the two dataframes
df = pd.concat([df1, df2], ignore_index=True)

# Convert the 'event_time' column to a datetime format
df['event_time'] = pd.to_datetime(df['event_time'])

# Calculate the RFM values
now = df['event_time'].max() + pd.DateOffset(days=1)
df['Recency'] = (now - df['event_time']).dt.days
df['Frequency'] = df.groupby('user_id')['user_id'].transform('count')
df['Monetary'] = df['price']
rfm = df.groupby('user_id').agg({'Recency': 'min', 'Frequency': 'max', 'Monetary': 'sum'})

# Define the segments based on the RFM values
segments = {
    'Best Customers': ['111', '211', '311', '411', '511', '611', '711', '811', '911', '1011'],
    'Loyal Customers': ['141', '142', '143', '241', '242', '243', '341', '342', '343', '441', '442', '443'],
    'Big Spenders': ['114', '124', '134', '144', '154', '164', '174', '184', '194', '1044'],
    'Almost Lost': ['411', '412', '413', '421', '422', '423', '431', '432', '433'],
    'Lost Customers': ['1411', '1421', '1431', '2411', '2421', '2431', '3411', '3421', '3431', '4411', '4421', '4431'],
    'Lost Cheap Customers': ['1141', '1142', '1143', '1241', '1242', '1243', '1341', '1342', '1343', '1441', '1442', '1443',
                             '1541', '1542', '1543', '1641', '1642', '1643', '1741', '1742', '1743', '1841', '1842', '1843',
                             '1941', '1942', '1943']
}

# Assign segment names to customers based on their RFM values
def get_segment(rfm, segments):
    rfm['RFM_Score'] = rfm['Recency'].astype(str) + rfm['Frequency'].astype(str) + rfm['Monetary'].astype(str)
    rfm['Segment'] = 'Other'
    for segment, score_range in segments.items():
        for score in score_range:
            if score in rfm['RFM_Score'].values:
                rfm.loc[rfm['RFM_Score'] == score, 'Segment'] = segment
    return rfm

rfm = get_segment(rfm, segments)

# Output the results
print(rfm.head())


segment_counts = rfm['Segment'].value_counts()
plt.bar(segment_counts.index, segment_counts.values)
plt.title('Distribution of Customers by Segment')
plt.xlabel('Segment')
plt.ylabel('Number of Customers')
plt.show()

segment_means = rfm.groupby('Segment').mean()[['Recency', 'Frequency', 'Monetary']]
print(segment_means)


# Define the colors for each segment
colors = {
    'Best Customers': 'green',
    'Loyal Customers': 'blue',
    'Big Spenders': 'orange',
    'Almost Lost': 'red',
    'Lost Customers': 'purple',
    'Lost Cheap Customers': 'gray',
    'Other': 'black'
}

# Define the axis labels and title
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.title('RFM Segments')

# Plot the scatter plot for each segment
for segment, color in colors.items():
    x = rfm.loc[rfm['Segment'] == segment, 'Recency']
    y = rfm.loc[rfm['Segment'] == segment, 'Frequency']
    plt.scatter(x, y, color=color, label=segment, alpha=0.5)

# Add the legend to the plot
plt.legend()

# Display the plot
plt.show()

import matplotlib.pyplot as plt

# Create a bar chart of the average RFM values for each segment
rfm_avg = rfm.groupby('Segment').agg({'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean'})
rfm_avg.plot(kind='bar')

# Add labels and title
plt.xlabel('Segment')
plt.ylabel('Average RFM Value')
plt.title('Average RFM Values by Segment')

# Show the plot
plt.show()



# Bar plot of RFM values by segment
rfm_bar = rfm.groupby('Segment').agg({'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean'})
rfm_bar.plot(kind='bar', figsize=(10, 6))
plt.title('Average RFM Values by Segment')
plt.xlabel('Segment')
plt.ylabel('RFM Values')
plt.show()


# Scatter plot of Monetary vs. Frequency, colored by segment
sns.scatterplot(data=rfm, x='Frequency', y='Monetary', hue='Segment', palette='colorblind')
plt.title('Monetary vs. Frequency by Segment')
plt.show()

import matplotlib.pyplot as plt

# Create the data
segments = ['Other']
recency = [18.310982]
frequency = [4.015258]
monetary = [1197.556308]

# Set the plot style
plt.style.use('ggplot')

# Create the figure and axis objects
fig, ax = plt.subplots()

# Plot the data
ax.bar(segments, recency, color='red', label='Recency')
ax.bar(segments, frequency, color='blue', bottom=recency, label='Frequency')
ax.bar(segments, monetary, color='green', bottom=[i+j for i,j in zip(recency,frequency)], label='Monetary')

# Add labels and title
ax.set_xlabel('Customer Segment')
ax.set_ylabel('Value')
ax.set_title('RFM Analysis by Segment')

# Add legend
ax.legend()

# Display the plot
plt.show()


