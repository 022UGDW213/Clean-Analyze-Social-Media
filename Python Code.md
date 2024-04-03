import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random


# Define the categories
categories = ['Food', 'Travel', 'Fashion', 'Fitness', 'Music', 'Culture', 'Family', 'Health']

# Generate random data dictionary
n = 500
data = {
    'Date': pd.date_range('2021-01-01', periods=n),
    'Category': [random.choice(categories) for _ in range(n)],
    'Likes': np.random.randint(0, 10000, size=n)
}

# Creating a DataFrame from the generated data
df = pd.DataFrame(data)
print(df.head())  # Display the first few rows of the generated data


# Assuming 'data' contains the randomly generated data dictionary

# Load data into a Pandas DataFrame
df = pd.DataFrame(data)

# Print the first few rows (head) of the DataFrame
print("DataFrame Head:")
print(df.head())

# Display DataFrame information
print("\nDataFrame Information:")
print(df.info())

# Descriptive statistics of the DataFrame
print("\nDataFrame Description:")
print(df.describe())

# Count of each 'Category' element
category_count = df['Category'].value_counts()
print("\nCount of each 'Category' element:")
print(category_count)


# Remove null values
df.dropna(inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Convert 'Date' field to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Convert 'Likes' data to integer
df['Likes'] = df['Likes'].astype(int)

# Displaying the cleaned DataFrame
print("Cleaned DataFrame:")
print(df.head())


# Visualizing Likes data using a histogram
plt.figure(figsize=(8, 6))
sns.histplot(df['Likes'], kde=False)
plt.xlabel('Likes')
plt.title('Distribution of Likes')
plt.show()


# Creating a boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Category', y='Likes', data=df)
plt.xlabel('Category')
plt.ylabel('Likes')
plt.title('Likes Distribution by Category')
plt.xticks(rotation=45)
plt.show()

# Computing statistics
# Mean of 'Likes' category
likes_mean = df['Likes'].mean()
print(f"\nMean of 'Likes': {likes_mean:.2f}")

# Mean of 'Likes' for each Category
category_likes_mean = df.groupby('Category')['Likes'].mean()
print("\nMean Likes for each Category:")
print(category_likes_mean)

