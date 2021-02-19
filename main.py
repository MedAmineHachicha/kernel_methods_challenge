from models import LogisticRegression, SVMClassifier
from utils import read_data, export_predictions, read_data_pandas, preprocess_tfidf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


X_train_paths = ['data/Xtr0_mat100.csv', 'data/Xtr1_mat100.csv', 'data/Xtr2_mat100.csv']
X_test_paths = ['data/Xte0_mat100.csv', 'data/Xte1_mat100.csv', 'data/Xte2_mat100.csv']
y_train_paths = ['data/Ytr0.csv', 'data/Ytr1.csv', 'data/Ytr2.csv']


X_train = read_data(paths=X_train_paths, delimiter=' ', skip_header=0)
X_test = read_data(paths=X_test_paths, delimiter=' ', skip_header=0)
y_train = read_data(paths=y_train_paths, delimiter=',', skip_header=1)

y_train = y_train[:, 1]


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, random_state=111)

clf = LogisticRegression()
clf.fit(X=X_train, y=y_train, max_iter=100000, eps=5e-5)
y_pred = clf.predict(X=X_val)
print('Training Accuracy: ', clf.get_accuracy_score(X_train, y_train))
print('Validation Accuracy: ', (y_pred == y_val).mean())

export_predictions(X_test, model=clf, filename='outputs/LogRegPred.csv')

gamma = X_train.shape[1]
clf = SVMClassifier(gamma=gamma, C=0.1)
clf.fit(X_train[:100], y_train[:100])
y_pred = clf.predict(X_val)
print('Validation Accuracy: ', (y_pred == y_val).mean())

export_predictions(X_test, model=clf, filename='outputs/SVMGauss.csv')



############ Using Raw Sequences #############
X_train_paths = ['data/Xtr0.csv', 'data/Xtr1.csv', 'data/Xtr2.csv']
X_test_paths = ['data/Xte0.csv', 'data/Xte1.csv', 'data/Xte2.csv']
y_train_paths = ['data/Ytr0.csv', 'data/Ytr1.csv', 'data/Ytr2.csv']

X_train = read_data_pandas(paths=X_train_paths, delimiter=',')
X_test = read_data_pandas(paths=X_test_paths, delimiter=',')
y_train = read_data_pandas(paths=y_train_paths, delimiter=',')
y_train = y_train['Bound'].values.astype(float)

tfidf_train, tfidf_test = preprocess_tfidf(X_train, X_test)
tfidf_train, tfidf_val, y_train, y_val = train_test_split(tfidf_train, y_train,
                                                          test_size=0.2, shuffle=True, random_state=111)

param_grid = {'C':[0.1, 1,10,100],'gamma':[1,0.1,0.001, 'auto', 'scale'], 'kernel':['rbf']}
grid = GridSearchCV(SVC(),param_grid,refit = True, verbose=2, n_jobs=-1, scoring='accuracy', cv=4)
grid.fit(tfidf_train,y_train['Bound'])

best_params = grid.best_params_


gamma = 0.1
clf = SVMClassifier(gamma=gamma, C=1)
clf.fit(tfidf_train, y_train)
y_pred = clf.predict(tfidf_val)
print('Validation Accuracy: ', (y_pred == y_val).mean())

export_predictions(X_test, model=clf, filename='outputs/SVMGauss.csv')


from sklearn.svm import SVC
clf = SVC()
clf.fit(tfidf_train, y_train)
y_pred = clf.predict(tfidf_val)
(y_pred == y_val).mean()


clf = LogisticRegression()
clf.fit(X=tfidf_train, y=y_train, max_iter=100000, eps=5e-5)
y_pred = clf.predict(X=tfidf_val)
print('Training Accuracy: ', clf.get_accuracy_score(tfidf_train, y_train))
print('Validation Accuracy: ', (y_pred == y_val).mean())

