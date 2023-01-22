import math
import re
import pprint
from typing import List, Mapping
from collections import Counter as ctr
from unicodedata import numeric

from scipy.sparse import csr_matrix
import itertools as it

import operator as op

from tqdm import tqdm

import pickle

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

def calc_idf(vocab: Mapping[str, int], dataset: List[List[str]], * , prec:int = 2):
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
            
            # 1 Find the Sparse Features 
            # ie sparse features for document {doc} = need to find common words between document & features in vocab
            words = sent_to_words(doc)
            unique_words = set(words)
            common_words = vocab.keys() & unique_words # sparse features

            # 2 calculate the TF ie term frequency for sparse words in document
            tf = {w: words.count(w) for w in common_words}

            # 3 Calculate TfIdf ie word -> tf*idf
            tfidf = {w: tf[w] * idfs.get(w, 0) for w in tf}

            # 3 Calculate L2 norms  (tfidf.values -> will hold all sparse values for sparse vector)
            l2_norm = math.sqrt(sum([x**2 for x in tfidf.values()]))

            # Prepare Sparse Vector Repr for {doc}
            for w in tf:
                rows.append(row)                 # row number
                columns.append(vocab.get(w, -1)) # column number
                vals.append(tfidf[w]/l2_norm)    # tfidf val append

        return csr_matrix((vals, (rows,columns)), shape=(len(dataset),len(vocab)))


def get_top(n, vocab, idfs):
    '''
    calculates top n features & corresponding vocab & idfs 
    (features are considered top with increasing val of their idf values)

    : param: n :- top n words to pick s.t. theirs idfs are high
    : param: vocab :- vocabulary of words to their column numbers
    : param: idfs :- mapping of words to their idf values

    :return tuple:- ( features, vocab, idfs )
    '''
    top_50_words = sorted(vocab, key=idfs.__getitem__, reverse=True)[:50]
    new_vocab = {w:i for i,w in enumerate(top_50_words)}
    new_idfs = {w:idfs[w] for w in top_50_words}

    return top_50_words, new_vocab, new_idfs


# if __name__ == '__main__':
#     d = {1:1, 2:2, 3:3, 4:40, 5:20, 6:10, 7:11, 8:7, 9:19, 10:5}
#     v = range(1,11)
#     t = sorted(v, key=d.__getitem__, reverse=True)[:7]
#     print(t)

if __name__ == '__main__':
    # corpus = [
    #  'this is the first document',
    #  'this document is the second document',
    #  'and this is the third one',
    #  'is this the first document',
    # ]
    ppath = r'real assignments/3. NLP/'
    filename = r'cleaned_strings'
    corpus = []
    with open(f'{ppath}{filename}', 'rb') as f:
        corpus = pickle.load(f)

    pp = pprint.PrettyPrinter()

    vocab, feature_names = fit(corpus)
    #pp.pprint(f'All features : {feature_names}')  

    idfs = calc_idf(vocab, corpus, prec=6)
    #pp.pprint(idfs)

    # get top 50 features

    features, vocab2, idfs2 = get_top(50, vocab, idfs)

    #pp.pprint(f'Top 50 fetures : {feature_names}')

    sparse_vec_repr = transform(corpus, vocab2, idfs2).toarray()

    #pp.pprint(sparse_vec_repr[0])

    with open(f'{ppath}cleaned_strings_all_features.txt', 'w') as f1, \
         open(f'{ppath}cleaned_strings_top_features.txt', 'w') as f2, \
         open(f'{ppath}cleaned_strings_first_vec_tfidf.txt', 'w') as f3, \
         open(f'{ppath}cleaned_strings_first_idf_original.txt', 'w') as f4:
        f1.write('\n'.join(feature_names))
        f2.writelines('\n'.join(features))
        f3.write(str(sparse_vec_repr[0]))
        f4.write(str(idfs))

    print(corpus[0])







