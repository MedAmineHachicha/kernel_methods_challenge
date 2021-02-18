from models import LogisticRegression, SVMClassifier
from utils import read_data, export_predictions

X_train_paths = ['data/Xtr0_mat100.csv', 'data/Xtr1_mat100.csv', 'data/Xtr2_mat100.csv']
X_test_paths = ['data/Xte0_mat100.csv', 'data/Xte1_mat100.csv', 'data/Xte2_mat100.csv']
y_train_paths = ['data/Ytr0.csv', 'data/Ytr1.csv', 'data/Ytr2.csv']


X_train = read_data(paths=X_train_paths, delimiter=' ', skip_header=0)
X_test = read_data(paths=X_test_paths, delimiter=' ', skip_header=0)
y_train = read_data(paths=y_train_paths, delimiter=',', skip_header=1)

y_train = y_train[:, 1]


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, random_state=111)

clf = LogisticRegression()
clf.fit(X=X_train, y=y_train, max_iter=100000, eps=5e-5)
y_pred = clf.predict(X=X_val)
print('Training Accuracy: ', clf.get_accuracy_score(X_train, y_train))
print('Validation Accuracy: ', (y_pred == y_val).mean())

export_predictions(X_test, model=clf, filename='outputs/LogRegPred.csv')

gamma = X_train.shape[1]/2
clf = SVMClassifier(gamma=gamma)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
print('Validation Accuracy: ', (y_pred == y_val).mean())

export_predictions(X_test, model=clf, filename='outputs/SVMGauss.csv')