import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def read_data(paths, delimiter, skip_header):
    data = []
    for path in paths:
        subdata = np.genfromtxt(path, delimiter=delimiter, skip_header=skip_header)
        data.append(subdata)
    data = np.concatenate(data)
    return data


def read_data_pandas(paths, delimiter):
    data = []
    for path in paths:
        subdata = pd.read_csv(path, delimiter=delimiter)
        data.append(subdata)
    data = pd.concat(data)
    return data


def export_predictions(X, model, filename):
    y_pred = model.predict(X)
    ids = np.array(range(3000))
    df = {'Id': ids,
          'Bound': y_pred.astype(int)}
    df = pd.DataFrame(df).set_index('Id')
    df.to_csv(filename)


def getKmers(sequence, size):
    return [sequence[x:x + size].lower() for x in range(len(sequence) - size + 1)]


def preprocess_tfidf(X_train, X_test):
    n1 = X_train.shape[0]
    X = pd.concat([X_train, X_test])
    X['kMers'] = X['seq'].apply(lambda x: getKmers(x, size=6))
    X['sentences'] = X['kMers'].apply(lambda x: ' '.join(x))

    vectorizer = TfidfVectorizer(max_df=0.95, min_df=5)
    tfidf = vectorizer.fit_transform(X['sentences']).toarray()

    pca = PCA(n_components=100)
    tfidf_ = pca.fit_transform(tfidf)

    scaler = StandardScaler()
    tfidf_ = scaler.fit_transform(tfidf_)

    return tfidf_[:n1], tfidf_[n1:]

