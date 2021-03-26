from itertools import product
import numpy as np


# Extracts all the possible kMers in a sequence
ngrams = lambda a, n: list(zip(*[a[i:] for i in range(n)]))


def Combinations(proteins, n):
    return list(product(proteins, repeat=n))


def get_spectrum_embeddings(Seq, combinations, n):
    kmers = ngrams(Seq, n)
    embedding = np.zeros(len(combinations))
    for ngram in kmers:
        index = combinations.index(ngram)
        embedding[index] += 1
    return embedding


def get_mismatch_embeddings(Seq, combinations, n):
    proteins = ['A', 'C', 'G', 'T']
    decompose_seq = ngrams(Seq, n)
    embedding = np.zeros(len(combinations))
    for kmer in decompose_seq:
        index = combinations.index(kmer)
        embedding[index] += 1
        kmer_seq = list(kmer)
        for ind, cur_protein in enumerate(kmer_seq):
            for protein in proteins:
                if protein != cur_protein:
                    mismatch_kmer = list(kmer_seq)
                    mismatch_kmer[ind] = protein
                    mismatch_kmer = tuple(mismatch_kmer)
                    index_ = combinations.index(mismatch_kmer)
                    embedding[index_] += 0.3
    return embedding

def get_gram_matrix(X1, X2=[]):

    n2 = len(X2)
    n1 = len(X1)
    if n2 == 0:
        gram_matrix = X1 @ X1.T
        gram_matrix_copy = X1 @ X1.T
        gram_matrix = gram_matrix.astype(np.float32)
        for i in range(n1):
            for j in range(n1):
                gram_matrix[i, j] /= (gram_matrix_copy[i, i] * gram_matrix_copy[j, j]) ** 0.5
        print('Gram Matrix Computed for X1')
        return gram_matrix
    else:
        gram_matrix = X1 @ X2.T
        gram_matrix = gram_matrix.astype(np.float32)
        gram_X1 = X1 @ X1.T
        gram_X2 = X2 @ X2.T

        for i in range(n1):
            for j in range(n2):
                gram_matrix[i, j] /= (gram_X2[j, j] * gram_X1[i, i]) ** 0.5
        print('Gram Matrix Computed for X2')
        return gram_matrix