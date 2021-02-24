from models import LogisticRegression, SVMClassifier
from utils import read_data, export_predictions, read_data_pandas, preprocess_tfidf, preprocess_TDA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np
import argparse


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


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("feature_engineering", help="Feature engineering technique", default='tfidf')
  parser.add_argument('model_used', help='The model used to generate prediction', default='SVM')
  args = parser.parse_args()
  main(args.feature_engineering, args.model_used)
  