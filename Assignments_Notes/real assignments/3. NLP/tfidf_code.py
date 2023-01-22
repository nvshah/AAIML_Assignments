import math
import re
import pprint
from typing import List, Mapping
from collections import Counter as ctr
from unicodedata import numeric

from scipy.sparse import csr_matrix
import itertools as it

from tqdm import tqdm

'''
# tqdm is a library that helps us to visualize the runtime of for loop. refer this to know more about tqdm
#https://tqdm.github.io/
'''

'''
1) Vocab ; (From Fit())
    dictionary where keys = words & values = column number
    Inorder to support Sparse Matrix Repr We are storing as Key-value Pair else
    We can use List[list] ie Matrix

2) Sparse Matrix (from Transform())
    list of Sparse Vectors for corpus's documents
'''

def sent_to_words(sentence):
    '''
        Convert sentence to List of Words
    '''
    #!! Currently using simple space to split words but can be enhanced with regex to 
    #TODO clean more nuance
    pattern = r"\s"
    w1 = re.split(pattern, sentence)
    w2 = [w for w in w1 if len(w1) > 2]
    return w2

def fit(dataset):
    '''
    Generate Vocab from given Corpus (ie collection of documents)
    : return: dict :- Word -> Column-num (For sparse Matrix)
    '''
    if isinstance(dataset, (list,)):
        unique_words = {w for s in dataset for w in sent_to_words(s)}
        words_list = sorted(unique_words)
        vocab = {j:i for i,j in enumerate(words_list)}
        
        return vocab, words_list
    else:
        print("you need to pass list of sentance")

def calc_idf(vocab: Mapping[str, int], dataset: List[List[str]], * , prec = 2):
    '''
    calculate the idf ie Inverse Document Frequency from given vocab & dataset
    : param: vocab :- vocabulary
    : param: dataset :- list of documents
    : prec: int :- precision for floating val of idf

    : return: float :- idf value mapping for each word in vocabulary
    '''
    doc_cnts = len(dataset)
    log_doc_cnts = math.log1p(doc_cnts)
    idf = dict()
    # creating the list of words for each doc
    corpus = [sent_to_words(d) for d in dataset] 

    for word in vocab:
        # number of documents with term t in it
        t_in_doc_cnt = sum([word in doc for doc in corpus]) 
        idf_t = 1 + log_doc_cnts - math.log1p(t_in_doc_cnt) #IDF(word)
        idf[word] = round(idf_t, prec)
    return idf 

def transform(dataset: List[List[str]], vocab: Mapping[str, int], idfs: Mapping[str, int]):
    '''
    Transform each documents in list to a new Sparse Vector repr for given vocabulary

    : param: dataset :- list of documents
    : param: vocab :- mapping of words to its coumn number in sparse matrix repr
    : param: idfs :- mapping of words to its idf values

    : return: list of sparse vectors corresp to each document
    
    '''
    '''(r, c, v) where 
      r = row num 
      c = col num 
      v = tfidf vals'''
    rows, columns, vals = [], [], []

    if isinstance(dataset, (list,)): # check if dataset is `List of List`
        for row, doc in enumerate(tqdm(dataset)):

            # 1 calculate the TF ie term frequency for words in document
            tf = ctr(sent_to_words(doc))

            # 2 Calculate TfIdf ie word -> tf*idf
            tfidf = {w: tf[w] * idfs[w] for w in tf}

            # 3 Calculate L2 norms  (tfidf.values -> will hold all sparse values for sparse vector)
            l2_norm = math.sqrt(sum([x**2 for x in tfidf.values()]))

            # Prepare Sparse Vector Repr for {doc}
            for w in tf:
                rows.append(row)         # row number
                columns.append(vocab[w]) # column number
                vals.append(tfidf[w]/l2_norm)    # tfidf val append

        return csr_matrix((vals, (rows,columns)), shape=(len(dataset),len(vocab)))

if __name__ == '__main__':
    corpus = [
     'this is the first document',
     'this document is the second document',
     'and this is the third one',
     'is this the first document',
    ]

    pp = pprint.PrettyPrinter()

    vocab, feature_names = fit(corpus)
    pp.pprint(feature_names)  

    idfs = calc_idf(vocab, corpus, prec=6)
    pp.pprint(idfs)

    sparse_vec_repr = transform(corpus, vocab, idfs).toarray()

    pp.pprint(sparse_vec_repr[0])




