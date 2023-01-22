# it will take classifier and set of values for hyper prameter in dict type dict({hyper parmeter: [list of values]})
# we are implementing this only for KNN, the hyper parameter should n_neighbors
from sklearn.metrics import accuracy_score

import numpy
from tqdm import tqdm
import numpy as np
import random

#---const----

KNN = "KNN"
PARAM_CNT = 10  # Ideally 10 is used for K-Fold Cross Validation

def randomly_select_60_percent_indices_in_range_from_1_to_len(x_train):
    return random.sample(range(0, len(x_train)), int(0.6*len(x_train)))


def randomSearch(x_train, y_train, classifier, params_range, folds):
    '''
    :param x_train: data points
    :param y_train: label points
    :param classifier: classifier model used to train data
    :param params_range: boundary for param val  
                         | format :- tuple (l, u) // lower & uper bound { inclusive both }
                         | (eg K val lower & uper bound for KNN classification)
    :param folds: number of folds to perform (ie number of buckets to divide data in)
    '''
    train_scores = []
    test_scores = []

    # As x_train & y_train have equal number of rows
    size = len(y_train)

    # 1. get random values for K
    l, u = params_range
    assert l < u, "l must be smaller than u"
    d = u-l+1
    k = d if d < PARAM_CNT else PARAM_CNT
    # k vals = hyper - params in case of KNN
    hyper_params = random.sample(range(l, u+1), k) # pick k random uniform values from range [l, u]

    # HYPER-PaRAMS are fixed before we start training a model

    # 2. Bucketizing ie Creating groups
    groups = np.array_split(range(size) ,folds)
    grp_size = len(groups)

    # HYPER_PARAMS
    for k in tqdm(hyper_params):
        #!info indices -> fold number & val -> metric found in that fold
        train_scores_folds = []  # train scores for different folds ie grps combination
        test_scores_folds = []  # test scores for different folds ie grp combinations
        
        # CURRENT {k} ENTER
        
        # FOLDS
        for i in range(grp_size):  
            # 3. Find the test & train indices for current fold
            test_idx = groups[i]
            # chain the indexes into single group
            train_idx = [idx for gi in range(grp_size) if gi != i for idx in groups[gi]]

            '''
                # NOTE :- when pass arr[[idx1, idx2, idx3]] 
                # in case of n-dimen array ie with multiple axis
                # it will first flaten/ravel the matrix & then 
                # select val present at 3 idxes ie {idx1, idx2, idx3} from single vector
            '''
            # selecting the data points based on the train_indices and test_indices
            X_train = x_train[train_idx]
            Y_train = y_train[train_idx]
            X_test  = x_train[test_idx]
            Y_test  = y_train[test_idx]

            # 4. Train Model for current fold {i} & hyper-param {k} 
            classifier.n_neighbors = k
            classifier.fit(X_train,Y_train)

            Y_predicted = classifier.predict(X_test)
            test_scores_folds.append(accuracy_score(Y_test, Y_predicted))

            Y_predicted = classifier.predict(X_train)
            train_scores_folds.append(accuracy_score(Y_train, Y_predicted))
        
        # Current {k} EXIT
        # at the end of each hyper-param calc append result to final result list 

        #! score will be avg val found for all folds
        train_scores.append(np.mean(np.array(train_scores_folds)))
        test_scores.append(np.mean(np.array(test_scores_folds)))
        
    return train_scores, test_scores

def GridSearch(x_train,y_train,classifier, params, folds):
    trainscores = []
    testscores  = []    
    for k in tqdm(params['n_neighbors']):
        trainscores_folds = []
        testscores_folds  = []
        for j in range(0, folds):
            # check this out: https://stackoverflow.com/a/9755548/4084039
            train_indices = randomly_select_60_percent_indices_in_range_from_1_to_len(x_train)
            test_indices  = list(set(list(range(1, len(x_train)))) - set(train_indices))

            # selecting the data points based on the train_indices and test_indices
            X_train = x_train[train_indices]
            Y_train = y_train[train_indices]
            X_test  = x_train[test_indices]
            Y_test  = y_train[test_indices]

            classifier.n_neighbors = k
            classifier.fit(X_train,Y_train)

            Y_predicted = classifier.predict(X_test)
            testscores_folds.append(accuracy_score(Y_test, Y_predicted))

            Y_predicted = classifier.predict(X_train)
            trainscores_folds.append(accuracy_score(Y_train, Y_predicted))
        trainscores.append(np.mean(np.array(trainscores_folds)))
        testscores.append(np.mean(np.array(testscores_folds)))
    return trainscores,testscores