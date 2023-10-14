import pandas as pd
import numpy as np
import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation    
sns.set(color_codes=True)
pd.set_option('display.max_columns', None)

from sklearn.model_selection import train_test_split

#Initial dataset (v1)
data_v1 = pd.read_csv("./athletes.csv")

# 1. Removing irrelevant columns and handling NaN values 
relevant_columns = ['region', 'age', 'weight', 'height', 'howlong', 'gender', 'eat', 'train',
                    'background', 'experience', 'schedule', 'deadlift', 'candj', 'snatch', 'backsq']
data_v2 = data_v1[relevant_columns].dropna()


# 2. Removing outliers

#Filter by weight
data_v2 = data_v2[data_v2['weight'] < 1500]

#Filter by gender
data_v2 = data_v2[data_v2['gender'] != '--']

#Filter by age
data_v2 = data_v2[data_v2['age'] >= 18]

#Filter by height
data_v2 = data_v2[(data_v2['height'] < 96) & (data_v2['height'] > 48)]

#Filter by deadlift 
male_deadlift_condition = (data_v2['gender'] == 'Male') & (data_v2['deadlift'] > 0) & (data_v2['deadlift'] <= 1105)
female_deadlift_conditions = (data_v2['gender'] == 'Female') & (data_v2['deadlift'] > 0) & (data_v2['deadlift'] <= 636)
data_v2 = data_v2[male_deadlift_condition | female_deadlift_conditions]

#Filter by candj
data_v2 = data_v2[(data_v2['candj'] > 0) & (data_v2['candj'] <= 395)]

#Filter by snatch
data_v2 = data_v2[(data_v2['snatch'] > 0) & (data_v2['snatch'] <= 496)]

#Filter by backsq
data_v2 = data_v2[(data_v2['backsq'] > 0) & (data_v2['backsq'] <= 1069)]

# 3. Cleaning survey data
data_v2 = data_v2.replace({'Decline to answer|': np.nan})
data_v2 = data_v2.dropna(subset=['background', 'experience', 'schedule', 'howlong', 'eat'])

numeric_columns = ['age', 'weight', 'height', 'howlong', 'deadlift', 'candj', 'snatch', 'backsq']

# 4. Determine the highest variability in columns to calculate total_lift

#Calculate standard deviation
data_v1_std_devs = data_v1[numeric_columns].std()
data_v2_std_devs = data_v2[numeric_columns].std()

#Sort columns by their standard devation
sorted_columns_v1 = data_v1_std_devs.sort_values(ascending=False).index.tolist()
sorted_columns_v2 = data_v2_std_devs.sort_values(ascending=False).index.tolist()

# print(sorted_columns_v1) # deadlift, backsq, snatch, candj
# print(sorted_columns_v2) # 'deadlift', 'backsq', 'candj', 'snatch'


# For data version 1
data_v1['total_lift'] = data_v1['deadlift'] + data_v1['candj'] + data_v1['snatch'] + data_v1['backsq']

#Split data for data_v1
X_v1  = data_v1.drop('total_lift', axis=1)
y_v1 = data_v1['total_lift']
X_train_v1, X_test_v1, y_train_v1, y_test_v1 = train_test_split(X_v1, y_v1, test_size=0.2, random_state=42) 


# For data version 2
data_v2['total_lift'] = data_v2['deadlift'] + data_v2['candj'] + data_v2['snatch'] + data_v2['backsq']

#Split data for data_v2
X_v2  = data_v2.drop('total_lift', axis=1)
y_v2 = data_v2['total_lift']
X_train_v2, X_test_v2, y_train_v2, y_test_v2 = train_test_split(X_v2, y_v2, test_size=0.2, random_state=42) 


# Save this cleaned dataset as v1
data_v1.to_csv('athletes.csv', index=False)

# Save this cleaned dataset as v2
data_v2.to_csv('athletes_v2.csv', index=False)
