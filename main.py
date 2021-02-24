from models import LogisticRegression, SVMClassifier
from utils import read_data, export_predictions, read_data_pandas, preprocess_tfidf, preprocess_TDA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np


def main(feature_engineering='tfidf', model_used='SVM'):

  if feature_engineering == 'tfidf':
      X_train_paths = ['data/Xtr0.csv', 'data/Xtr1.csv', 'data/Xtr2.csv']
      X_test_paths = ['data/Xte0.csv', 'data/Xte1.csv', 'data/Xte2.csv']
      y_train_paths = ['data/Ytr0.csv', 'data/Ytr1.csv', 'data/Ytr2.csv']

      X_train = read_data_pandas(paths=X_train_paths, delimiter=',')
      X_test = read_data_pandas(paths=X_test_paths, delimiter=',')
      y_train = read_data_pandas(paths=y_train_paths, delimiter=',')
      y_train = y_train['Bound'].values.astype(float)

      tfidf_train, tfidf_test = preprocess_tfidf(X_train, X_test,
                                                 window_size=6,
                                                 pca_components=100)
      X_train, X_val, y_train, y_val = train_test_split(tfidf_train, y_train,
                                                                test_size=0.2, shuffle=True,
                                                                 random_state=111)
  elif feature_engineering == 'TDA':
    
    X_train_paths = ['data/Xtr0.csv', 'data/Xtr1.csv', 'data/Xtr2.csv']
    X_test_paths = ['data/Xte0.csv', 'data/Xte1.csv', 'data/Xte2.csv']
    y_train_paths = ['data/Ytr0.csv', 'data/Ytr1.csv', 'data/Ytr2.csv']

    X_train = read_data_pandas(paths=X_train_paths, delimiter=',')
    X_test = read_data_pandas(paths=X_test_paths, delimiter=',')
    y_train = read_data_pandas(paths=y_train_paths, delimiter=',')
    y_train = y_train['Bound'].values.astype(float)

    train_dgms, test_dgms = preprocess_TDA(X_train, X_test)
    print('Train/val split')
    X_train, X_val, y_train, y_val = train_test_split(train_dgms, y_train,
                                                      test_size=0.2, shuffle=True, 
                                                      random_state=111)
    

  elif feature_engineering == 'kmeans':
      X_train_paths = ['data/Xtr0_mat100.csv', 'data/Xtr1_mat100.csv', 'data/Xtr2_mat100.csv']
      X_test_paths = ['data/Xte0_mat100.csv', 'data/Xte1_mat100.csv', 'data/Xte2_mat100.csv']
      y_train_paths = ['data/Ytr0.csv', 'data/Ytr1.csv', 'data/Ytr2.csv']

      X_train = read_data(paths=X_train_paths, delimiter=' ', skip_header=0)
      X_test = read_data(paths=X_test_paths, delimiter=' ', skip_header=0)
      y_train = read_data(paths=y_train_paths, delimiter=',', skip_header=1)
      y_train = y_train[:, 1]

      X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, random_state=111)

  if model_used == 'LogReg':
      clf = LogisticRegression()
      clf.fit(X=X_train, y=y_train, max_iter=100000, eps=5e-5)
      y_pred = clf.predict(X=X_val)
      print('Training Accuracy: ', clf.get_accuracy_score(X_train, y_train))
      print('Validation Accuracy: ', (y_pred == y_val).mean())

      export_predictions(X_test, model=clf, filename='outputs/LogRegPred.csv')

  elif model_used == 'SVM'
      X = np.concatenate([X_train, X_val])
      y = np.concatenate([y_train, y_val])
      if data_used == 'raw':
        gamma = 1/ (X.shape[1] * X.var())
      else:
        gamma = X_train.shape[1]
      clf = SVMClassifier(gamma=gamma, C=0.1)
      clf.fit(X_train, y_train)
      y_pred = clf.predict(X_val)
      print('Validation Accuracy: ', (y_pred == y_val).mean())

      export_predictions(X_test, model=clf, filename='outputs/SVMGauss.csv')
  elif model_used == 'TDA + SVM':

    # Definition of pipeline
    from sklearn.preprocessing   import MinMaxScaler
    from sklearn.pipeline        import Pipeline
    from sklearn.svm             import SVC
    pipe = Pipeline([("Separator", gd.representations.DiagramSelector(limit=np.inf, point_type="finite")),
                    ("Scaler",    gd.representations.DiagramScaler(scalers=[([0,1], MinMaxScaler())])),
                    ("TDA",       gd.representations.PersistenceImage()),
                    ("Estimator", SVC())])
    param =    [{"Scaler__use":         [False],
                "TDA":                 [gd.representations.SlicedWassersteinKernel()], 
                "TDA__bandwidth":      [0.1, 1.0],
                "TDA__num_directions": [20],
                "Estimator":           [SVC(kernel="precomputed", gamma="auto")]},
              ]

    from sklearn.model_selection import GridSearchCV

    clf = GridSearchCV(pipe, param, cv=5) #maybe put cv=3 for faster results 
    clf = clf.fit(X_train, y_train)
    #print(model.best_params_)
    y_pred = clf.predict(X_val)
    print('Validation Accuracy: ', (y_pred == y_val).mean())
    export_predictions(test_dgms, model=clf, filename='outputs/SVM_TDA_kernel.csv')



############ Using Raw Sequences #############
# X_train_paths = ['data/Xtr0.csv', 'data/Xtr1.csv', 'data/Xtr2.csv']
# X_test_paths = ['data/Xte0.csv', 'data/Xte1.csv', 'data/Xte2.csv']
# y_train_paths = ['data/Ytr0.csv', 'data/Ytr1.csv', 'data/Ytr2.csv']

# X_train = read_data_pandas(paths=X_train_paths, delimiter=',')
# X_test = read_data_pandas(paths=X_test_paths, delimiter=',')
# y_train = read_data_pandas(paths=y_train_paths, delimiter=',')
# y_train = y_train['Bound'].values.astype(float)

# tfidf_train, tfidf_test = preprocess_tfidf(X_train, X_test, window_size=6, pca_components=100)
# tfidf_train, tfidf_val, y_train, y_val = train_test_split(tfidf_train, y_train,
#                                                           test_size=0.2, shuffle=True, random_state=111)


'''from sklearn.svm import SVC
clf1 = SVC(gamma='scale', kernel='rbf')
clf1.fit(tfidf_train, y_train)
y_pred = clf1.predict(tfidf_val)
(y_pred == y_val).mean()'''


# param_grid = {'C' : [0.1, 1,10,100],
#               'gamma' : [1,0.1,0.001,'auto', 'scale'],
#               'kernel' : ['rbf']}

# grid = GridSearchCV(SVC(),param_grid,refit = True, verbose=2, n_jobs=-1, scoring='accuracy', cv=4)
# grid.fit(tfidf_train,y_train)

# best_params = grid.best_params_

# tfidf = np.concatenate([tfidf_train, tfidf_val])
# y = np.concatenate([y_train, y_val])
# gamma = 1/ (tfidf.shape[1] * tfidf.var())
# clf = SVMClassifier(gamma=gamma, C=1)
# clf.fit(tfidf_train, y_train)
# y_pred = clf.predict(tfidf_val)
# print('Validation Accuracy: ', (y_pred == y_val).mean())

# clf = SVMClassifier(gamma=gamma, C=1)
# clf.fit(tfidf, y)
# export_predictions(tfidf_test, model=clf, filename='outputs/SVM_Gauss_tfidf.csv')



#### TDA kernel : to be tested ####

#!pip install gudhi 
# import gudhi as gd
# import gudhi.representations
# from utils import preprocess_TDA

# train_dgms, test_dgms = preprocess_TDA(X_train,X_test)
# print('Train/val split')
# tda_train, tda_val, y_train, y_val = train_test_split(train_dgms, y_train, test_size=0.2, shuffle=True, random_state=111)

# # Definition of pipeline
# from sklearn.preprocessing   import MinMaxScaler
# from sklearn.pipeline        import Pipeline
# from sklearn.svm             import SVC
# pipe = Pipeline([("Separator", gd.representations.DiagramSelector(limit=np.inf, point_type="finite")),
#                  ("Scaler",    gd.representations.DiagramScaler(scalers=[([0,1], MinMaxScaler())])),
#                  ("TDA",       gd.representations.PersistenceImage()),
#                  ("Estimator", SVC())])
# param =    [{"Scaler__use":         [False],
#              "TDA":                 [gd.representations.SlicedWassersteinKernel()], 
#              "TDA__bandwidth":      [0.1, 1.0],
#              "TDA__num_directions": [20],
#              "Estimator":           [SVC(kernel="precomputed", gamma="auto")]},
#            ]

# from sklearn.model_selection import GridSearchCV

# clf = GridSearchCV(pipe, param, cv=5) #maybe put cv=3 for faster results 
# clf = clf.fit(tda_train, y_train)
# #print(model.best_params_)
# y_pred = clf.predict(tda_val)
# print('Validation Accuracy: ', (y_pred == y_val).mean())
# export_predictions(test_dgms, model=clf, filename='outputs/SVM_TDA_kernel.csv')
