pip install pandas

Defaulting to user installation because normal site-packages is not writeable
Collecting pandas
  Downloading pandas-2.2.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (13.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 13.0/13.0 MB 186.8 kB/s eta 0:00:00m eta 0:00:01[36m0:00:02
Collecting numpy<2,>=1.22.4
  Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.2/18.2 MB 641.8 kB/s eta 0:00:00m eta 0:00:01[36m0:00:01
Requirement already satisfied: pytz>=2020.1 in /usr/lib/python3/dist-packages (from pandas) (2022.1)
Requirement already satisfied: python-dateutil>=2.8.2 in ./.local/lib/python3.10/site-packages (from pandas) (2.9.0.post0)
Collecting tzdata>=2022.7
  Downloading tzdata-2024.1-py2.py3-none-any.whl (345 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 345.4/345.4 KB 1.2 MB/s eta 0:00:00 MB/s eta 0:00:01:01
Requirement already satisfied: six>=1.5 in /usr/lib/python3/dist-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)
Installing collected packages: tzdata, numpy, pandas
Successfully installed numpy-1.26.4 pandas-2.2.1 tzdata-2024.1
Note: you may need to restart the kernel to use updated packages.



pip install matplotlib

Defaulting to user installation because normal site-packages is not writeable
Collecting matplotlib
  Downloading matplotlib-3.8.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (11.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.6/11.6 MB 1.5 MB/s eta 0:00:00m eta 0:00:01[36m0:00:01
Collecting contourpy>=1.0.1
  Downloading contourpy-1.2.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (305 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 305.2/305.2 KB 1.5 MB/s eta 0:00:00 MB/s eta 0:00:01
Collecting cycler>=0.10
  Downloading cycler-0.12.1-py3-none-any.whl (8.3 kB)
Requirement already satisfied: packaging>=20.0 in ./.local/lib/python3.10/site-packages (from matplotlib) (24.0)
Collecting fonttools>=4.22.0
  Downloading fonttools-4.50.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.6/4.6 MB 1.2 MB/s eta 0:00:00m eta 0:00:01[36m0:00:01
Collecting kiwisolver>=1.3.1
  Downloading kiwisolver-1.4.5-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (1.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/1.6 MB 1.1 MB/s eta 0:00:00[36m0:00:01[36m0:00:01:010m
Requirement already satisfied: python-dateutil>=2.7 in ./.local/lib/python3.10/site-packages (from matplotlib) (2.9.0.post0)
Requirement already satisfied: pyparsing>=2.3.1 in /usr/lib/python3/dist-packages (from matplotlib) (2.4.7)
Requirement already satisfied: numpy<2,>=1.21 in ./.local/lib/python3.10/site-packages (from matplotlib) (1.26.4)
Requirement already satisfied: pillow>=8 in /usr/lib/python3/dist-packages (from matplotlib) (9.0.1)
Requirement already satisfied: six>=1.5 in /usr/lib/python3/dist-packages (from python-dateutil>=2.7->matplotlib) (1.16.0)
Installing collected packages: kiwisolver, fonttools, cycler, contourpy, matplotlib
Successfully installed contourpy-1.2.1 cycler-0.12.1 fonttools-4.50.0 kiwisolver-1.4.5 matplotlib-3.8.3
Note: you may need to restart the kernel to use updated packages.


pip install seaborn

Defaulting to user installation because normal site-packages is not writeable
Collecting seaborn
  Downloading seaborn-0.13.2-py3-none-any.whl (294 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 294.9/294.9 KB 2.2 MB/s eta 0:00:00 MB/s eta 0:00:01:01
Requirement already satisfied: matplotlib!=3.6.1,>=3.4 in ./.local/lib/python3.10/site-packages (from seaborn) (3.8.3)
Requirement already satisfied: pandas>=1.2 in ./.local/lib/python3.10/site-packages (from seaborn) (2.2.1)
Requirement already satisfied: numpy!=1.24.0,>=1.20 in ./.local/lib/python3.10/site-packages (from seaborn) (1.26.4)
Requirement already satisfied: contourpy>=1.0.1 in ./.local/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (1.2.1)
Requirement already satisfied: python-dateutil>=2.7 in ./.local/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (2.9.0.post0)
Requirement already satisfied: cycler>=0.10 in ./.local/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (0.12.1)
Requirement already satisfied: pyparsing>=2.3.1 in /usr/lib/python3/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (2.4.7)
Requirement already satisfied: kiwisolver>=1.3.1 in ./.local/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (1.4.5)
Requirement already satisfied: packaging>=20.0 in ./.local/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (24.0)
Requirement already satisfied: pillow>=8 in /usr/lib/python3/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (9.0.1)
Requirement already satisfied: fonttools>=4.22.0 in ./.local/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (4.50.0)
Requirement already satisfied: pytz>=2020.1 in /usr/lib/python3/dist-packages (from pandas>=1.2->seaborn) (2022.1)
Requirement already satisfied: tzdata>=2022.7 in ./.local/lib/python3.10/site-packages (from pandas>=1.2->seaborn) (2024.1)
Requirement already satisfied: six>=1.5 in /usr/lib/python3/dist-packages (from python-dateutil>=2.7->matplotlib!=3.6.1,>=3.4->seaborn) (1.16.0)
Installing collected packages: seaborn
Successfully installed seaborn-0.13.2
Note: you may need to restart the kernel to use updated packages.


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

