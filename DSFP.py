import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor



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

# The baseline predictions are the historical averages
baseline_preds = test_features[:, column_list.index('serum_creatinine')]

# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2))

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels);

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
print('Errors:', errors)



