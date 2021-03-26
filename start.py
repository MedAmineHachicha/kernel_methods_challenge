import pandas as pd
import numpy as np
from models import SVM_custom_kernel
from tqdm import tqdm
from kernels import Combinations, get_mismatch_embeddings, get_spectrum_embeddings, get_gram_matrix
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import accuracy_score
import time


def get_dataset_predictions(dataset_index='0'):

    # Read dataset
    X_train = (pd.read_csv(f'data/Xtr{dataset_index}.csv',header=None).values).tolist()
    Y_train = (pd.read_csv(f'data/Ytr{dataset_index}.csv',sep=',',index_col=0).values)
    X_train = (np.array(X_train)[1:,1]).tolist()
    X_test = (pd.read_csv(f'data/Xte{dataset_index}.csv',header=None).values).tolist()
    X_test = (np.array(X_test)[1:,1]).tolist()
    Y_train[Y_train == 0] = -1

    kernels = {'mismatch_6': [6, get_mismatch_embeddings],
                 'mismatch_7': [7, get_mismatch_embeddings],
                'spectrum_7': [7, get_spectrum_embeddings],
                'spectrum_6': [6, get_spectrum_embeddings],
                'spectrum_8': [8, get_spectrum_embeddings],
                'mismatch_8': [8, get_mismatch_embeddings]}

    from os import path

    gram_matrices = {}
    for key in kernels.keys():
        train_filename = 'gram_matrices/train_' + key + f'_dataset{dataset_index}.npy'
        test_filename = 'gram_matrices/test_' + key + f'_dataset{dataset_index}.npy'
        length = kernels[key][0]
        embedding_func = kernels[key][1]
        DNA_combinations = Combinations(proteins=['A', 'C', 'G', 'T'], n=length)

        if path.exists(train_filename):
            print(train_filename, ' already exists !')
            gram_train = np.load(train_filename)
        else:
            print('Creating ', train_filename)
            train_embeddings = np.empty([len(X_train), len(DNA_combinations)])
            for i in tqdm(range(len(X_train))):
                train_embeddings[i, :] = embedding_func(Seq=X_train[i], combinations=DNA_combinations, n=length)
            gram_train = get_gram_matrix(train_embeddings)
            np.save(train_filename, gram_train)
        if path.exists(test_filename):
            print(test_filename, ' already exists !')
            gram_test = np.load(test_filename)
        else:
            print('Creating ', test_filename)
            test_embeddings = np.empty([len(X_test), len(DNA_combinations)])
            for i in tqdm(range(len(X_test))):
                test_embeddings[i, :] = embedding_func(Seq=X_test[i], combinations=DNA_combinations, n=length)
            gram_test = get_gram_matrix(train_embeddings, test_embeddings)
            np.save(test_filename, gram_test)
        gram_matrices[key] = {'train': gram_train,
                                'test': gram_test}


    C = 0.5
    eps = 1e-4
    list_kernels = list(gram_matrices.keys())
    list_predictions = []
    for i in range(3):
        gram_train = np.zeros((2000, 2000))
        gram_test = np.zeros((2000, 1000))
        sampled_kernels = random.sample(list_kernels, 5)
        for kernel in sampled_kernels:
            print(f'Using {kernel}')
            gram_train += gram_matrices[kernel]['train']
            gram_test += gram_matrices[kernel]['test']
        gram_train /= 5
        gram_test /= 5
        list_train, list_val = train_test_split(list(range(2000)), test_size=0.2)
        gram_train_split = gram_train[list_train, :][:, list_train]
        gram_val_split = gram_train[list_train, :][:, list_val]
        y_train_split = Y_train[list_train]
        y_val_split = Y_train[list_val]

        svm_test = SVM_custom_kernel(c=C, eps=eps)
        svm_test.fit(gram_train_split, y_train_split)
        y_val_pred = svm_test.predict_class(gram_val_split).reshape(-1)
        print('Val Accuracy =', accuracy_score(y_val_split.reshape(-1), y_val_pred))
        y_test_pred = svm_test.predict_class(gram_test[list_train, :])
        y_test_pred[y_test_pred == -1] = 0
        list_predictions.append(y_test_pred)

    y_pred = np.array(np.array(list_predictions).mean(axis=0).reshape((-1,))>0.5,dtype=int)

    return y_pred

if __name__== '__main__':
    print('To skip Gram matrices calculations, please clone the full Github repo with all files under gram_matrices folder')
    print('We are running SVM for 3 iterations before ensembling instead of 13 models for faster results')
    time.sleep(5)

    print('Processing and predicting for dataset 0')
    time.sleep(2)
    y_pred_0 = get_dataset_predictions(dataset_index='0')

    print('Processing and predicting for dataset 1')
    time.sleep(2)
    y_pred_1 = get_dataset_predictions(dataset_index='1')

    print('Processing and predicting for dataset 2')
    time.sleep(2)
    y_pred_2 = get_dataset_predictions(dataset_index='2')

    print('Exporting predictions')
    y_pred = list(y_pred_0) + list(y_pred_1) + list(y_pred_2)

    y_pred = pd.DataFrame(y_pred, columns=['Bound'])
    y_pred.index.name = 'Id'
    y_pred.to_csv('Yte.csv')
    print('Done')
