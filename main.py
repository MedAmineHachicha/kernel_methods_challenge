import pandas as pd

X1 = pd.read_csv('data/Xtr0_mat100.csv', sep=' ', header=None)
y1 = pd.read_csv('data/Ytr0.csv', sep=',')

X2 = pd.read_csv('data/Xtr1_mat100.csv', sep=' ', header=None)
y2 = pd.read_csv('data/Ytr1.csv', sep=',')

X3 = pd.read_csv('data/Xtr2_mat100.csv', sep=' ', header=None)
y3 = pd.read_csv('data/Ytr2.csv', sep=',')

X = pd.concat([X1, X2, X3])
y = pd.concat([y1, y2, y3])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=123)

from sklearn.svm import SVC

clf = SVC(C=10, kernel='linear')

clf.fit(X_train, y_train['Bound'])
y_pred = clf.predict(X_test)

(y_pred == y_test['Bound']).mean()


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

clf.fit(X_train, y_train['Bound'])
y_pred = clf.predict(X_test)
(y_pred == y_test['Bound']).mean()

from sklearn.linear_model import LogisticRegressionCV
clf = LogisticRegressionCV(Cs=5, cv=4)
clf.fit(X_train, y_train['Bound'])
y_pred = clf.predict(X_test)
(y_pred == y_test['Bound']).mean()

