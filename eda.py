import pandas as pd
import numpy as np
import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation    
sns.set(color_codes=True)
pd.set_option('display.max_columns', None)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data_v1 = pd.read_csv('athletes.csv')
data_v2 = pd.read_csv('athletes_v2.csv')



# # ----- EDA ON DATA_V1 -----
# check_types_v1 = data_v1.dtypes
# # print(check_types_v1)

# check_shape_v1 = data_v1.shape
# # print(check_shape_v1)

# check_count_v1 = data_v1.count()
# # print(check_count_v1)

# data_v1 = data_v1.drop_duplicates()
# print(data_v1.isnull().sum())

# basic_stats_v1 = data_v1.describe()
# print(basic_stats_v1)

# missing_values_v1 = data_v1.isnull().sum()
# print(missing_values_v1)

# #Histogram for age, weight and height
# data_v1[['age', 'weight', 'height']].hist(figsize=(10,8))
# plt.tight_layout()
# plt.show()

# #Boxplot for numeric lifting columns
# lift_columns_v1 = ['deadlift', 'candj', 'snatch', 'backsq']
# for col in lift_columns_v1:
#     plt.figure(figsize=(6,4))
#     sns.boxplot(y=data_v1[col])
#     plt.title(f'Boxplot of {col}')
#     plt.show()

# #Bar chart for categorical columns like 'gender', 'eat', 'train'
# categorical_columns_v1 = ['gender', 'eat', 'train']
# for col in categorical_columns_v1:
#     plt.figure(figsize=(6,4))
#     sns.boxplot(y=data_v2[col])
#     plt.title(f'Bar chart of {col}')
#     plt.show()

data_v1['total_lift'] = data_v1['deadlift'] + data_v1['candj'] + data_v1['snatch'] + data_v1['backsq']

X_v1  = data_v1.drop('total_lift', axis=1)
y_v1 = data_v1['total_lift']
X_train_v1, X_test_v1, y_train_v1, y_test_v1 = train_test_split(X_v1, y_v1, test_size=0.2, random_state=42) 


# Baseline model on data version 1
model_v1 = LinearRegression()
model_v1.fit(X_train_v1, y_train_v1)

