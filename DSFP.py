import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df = pd.read_csv('hfdataset.csv')
df = df.drop('time', 1)

# df1 = df['platelets']
# df2 = df['serum_creatinine']

def corr_matrix(df):
	f = plt.figure(figsize=(12, 7))
	plt.matshow(df.corr(), fignum=f.number)
	plt.xticks(range(df.shape[1]), df.columns, fontsize=6, rotation=45)
	plt.yticks(range(df.shape[1]), df.columns, fontsize=7)
	cb = plt.colorbar()
	cb.ax.tick_params(labelsize=5)
	plt.title("Correlation Matrix")
	plt.show()

# corr_matrix(df)
# print(df.describe())

# Labels are the values we want to predict
labels = np.array(df['DEATH_EVENT'])
# Remove the labels from the features
# axis 1 refers to the columns
df= df.drop('DEATH_EVENT', axis = 1)
# Saving feature names for later use
column_list = list(df.columns)
# Convert to numpy array
df = np.array(df)


train_features, test_features, train_labels, test_labels = train_test_split(df, labels, test_size = 0.2, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)




