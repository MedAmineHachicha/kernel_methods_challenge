import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler



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
    return [sequence[x:x + size] for x in range(len(sequence) - size + 1)]


def preprocess_tfidf(X_train, X_test, window_size=6, pca_components=100):
    n1 = X_train.shape[0]
    X = pd.concat([X_train, X_test])
    X['kMers'] = X['seq'].apply(lambda x: getKmers(x, size=window_size))
    X['sentences'] = X['kMers'].apply(lambda x: ' '.join(x))

    vectorizer = TfidfVectorizer(min_df=5)
    tfidf = vectorizer.fit_transform(X['sentences']).toarray()

    pca = PCA(n_components=pca_components)
    tfidf_ = pca.fit_transform(tfidf)

    scaler = MinMaxScaler()
    tfidf_ = scaler.fit_transform(tfidf_)

    return tfidf_[:n1], tfidf_[n1:]


import gudhi as gd
import gudhi.representations
from gudhi import plot_persistence_diagram 
import tqdm.notebook as tq


def preprocess_TDA(X_train,X_test,base2idx = {'A':0,'C':1,'G':2,'T':3},e = np.eye(4)):
    
    print('Preprocess training samples : Compute Persistence Diagrams..')
    train_dgms = compute_persistence_diagrams(X_train['seq'].values)
    print('Preprocess test samples : Compute Persistence Diagrams..')
    test_dgms = compute_persistence_diagrams(X_test['seq'].values)
    return train_dgms, test_dgms

def compute_persistence_diagrams(X):
    simplices = []
    for i,seq in tq.tqdm(enumerate(X),total = len(X)):
        dgm, simplex_tree = seq2pd(seq) #compute persistence diagrams but only use simplices for Sliced Wasserstein Kernel
        simplices.append(simplex_tree)
    return simplices

def seq2pd(seq):
    
    b = seq2point_cloud(seq)
    skeleton = gd.RipsComplex(points = b, max_edge_length = 0.8)
    Rips_simplex_tree_sample = skeleton.create_simplex_tree(max_dimension = 2)
    #compute persistence diagram for dimensions 0 and 1
    #but only 1-dimensional features will be used for the kernel
    dgm = Rips_simplex_tree_sample.persistence()
    return dgm, Rips_simplex_tree_sample

def seq2point_cloud(seq):
    b = np.zeros((len(seq),4))
    for k,base in enumerate(seq):
        if k==0:
            b[k] = e[base2idx[base]]
        else:
            b[k] = 0.5*(b[k-1]+e[base2idx[base]])
    return b

