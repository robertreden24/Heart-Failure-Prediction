import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import svm

df = pd.read_csv('hfdataset.csv')
df = df.drop('time', 1)

X = (df[['serum_creatinine','ejection_fraction']].to_numpy())
# X = df.to_numpy()
y = df['DEATH_EVENT'].to_list()

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=0)

model = svm.SVC(kernel='linear')
model.fit(train_X,train_y)


predict_death = model.predict(test_X)
print(confusion_matrix(test_y, predict_death))
print("Accuracy: ", accuracy_score(test_y, predict_death))
print(classification_report(test_y, predict_death))


plt.scatter(train_X[:,0], train_X[:,1], c=train_y, cmap='winter')
ax = plt.gca()
xlim = ax.get_xlim()

ax.scatter(test_X[:,0], test_X[:,1], c=test_y, cmap='winter')


w = model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(xlim[0], xlim[1])
yy = a * xx - (model.intercept_[0] / w[1])
plt.plot(xx,yy)

plt.xlabel('Serum Creatinine')
plt.ylabel('Ejection Fraction')

plt.show()