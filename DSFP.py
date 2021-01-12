import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score, classification_report

df = pd.read_csv('hfdataset.csv')
#drop the time column as it is not used
df = df.drop('time', 1)

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
df= df.drop('DEATH_EVENT', axis = 1)
column_list = list(df.columns)
df = np.array(df)

train_features, test_features, train_labels, test_labels = train_test_split(df, labels, test_size = 0.2, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

baseline_preds = test_features[:, column_list.index('serum_creatinine')]
# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2), "degrees.")

# # Instantiate model with 1000 decision trees
# rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# # Train the model on training data
# rf.fit(train_features, train_labels);

# # Use the forest's predict method on the test data
# predictions = rf.predict(test_features)
# #round up or down the prediction to nearest integer
# round_pred = []
# for n in predictions:
# 	x = round(n)
# 	round_pred.append(x)
# # Calculate the absolute errors
# errors = abs(predictions - test_labels)
# # Print out the mean absolute error (mae)
# print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# #MCC Evaluation
# print('MCC: ', matthews_corrcoef(test_labels, round_pred, sample_weight=None))

# # Get numerical feature importances
# importances = list(rf.feature_importances_)
# # List of tuples with variable and importance
# feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(column_list, importances)]
# # Sort the feature importances by most important first
# feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# # Print out the feature and importances 
# [print('Variable: {:25} Importance: {}'.format(*pair)) for pair in feature_importances];


# New random forest using only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)
# Extract the two most important features
important_indices = [column_list.index('serum_creatinine'), column_list.index('ejection_fraction')]
train_important = train_features[:, important_indices]
test_important = test_features[:, important_indices]
# Train the random forest
rf_most_important.fit(train_important, train_labels)
# Make predictions and determine the error
predictions = rf_most_important.predict(test_important)
errors = abs(predictions - test_labels)

# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# round up or down the prediction to nearest integer
round_pred = []
for n in predictions:
	x = round(n)
	round_pred.append(x)

print('MCC: ', matthews_corrcoef(test_labels, round_pred, sample_weight=None))
print(confusion_matrix(test_labels, round_pred))
print(classification_report(test_labels, round_pred))
print("Accuracy: ", accuracy_score(test_labels, round_pred)) 
