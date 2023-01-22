import numpy as np
import pandas as pd

# Actual True (Just for Understanding)
at = pd.Series(
    [0, 0],
    index=['pt', 'pf']
)
# Actual False
af = pd.Series(
    [0, 0],
    index=['pt', 'pf']
)

cm = pd.DataFrame({'at': at, 'af':af})
cm

# -----------------
# (Utils)

# 1. Confusion Matrix  
def calc_binary_confusion_matrix(d):
    '''
    :param d: dataframe with 2 series ie actual & predicted
    : return : numpy array
    '''
    #d = df[['y', 'y_pred']]
    tp = fp = fn = tn = 0
    for _, r in d.iterrows():
        ya, yp = r
        if ya == yp:
            if yp :  # True Positive
                tp += 1
            else:    # True Negative
                tn += 1
        else:
            if yp :  # False Positive
                fp += 1
            else:    # False Negative
                fn += 1

    ans = np.array([[tn, fn], [fp, tp]])
#     ans
#     cm = pd.DataFrame(ans)
    return ans

# 2. F1 Score
def calc_f1_score(d):
    '''
    :param d: dataframe with 2 series ie actual & predicted
    '''
    cm = calc_binary_confusion_matrix(d)
    tp, fp, fn = cm[1, 1], cm[1, 0], cm[0, 1]  # as we are using 1/3 of cells so calculated CM
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1_score = 2 / (recall**-1 + precision**-1)
    return f1_score

# 3. AUC Score

def get_tpr_fpr(d):
    '''
    :param d: dataframe with 2 series ie actual & predicted
    :return : tuple :- (tpr, fpr)
    '''
    y_actual = d.columns[0] # actual column name
    tp = fp = 0
    cnts = d[y_actual].value_counts()

    an, ap = cnts.get(0, 0), cnts.get(1, 0)
    for i, r in d.iterrows():

        ya, yp = r
        if ya == yp:
            if yp :  # True Positive
                tp += 1
        else:
            if yp :  # False Positive
                fp += 1

    # TODO decide What to do
    tpr = (tp / ap) if ap else 0
    fpr = (fp / an) if an else 0

    return tpr, fpr

def calc_auc_score(d):
    '''
    :param d: dataframe with 2 series ie actual & predicted
    :return : AUC score
    '''
    y_act, y_proba, *_ = d.columns
    
    # 1) Sort values by pred
    d_s = d.sort_values(by=y_proba, ascending=False)
    unique_proba = d_s[y_proba].unique()
    tpr_array = []
    fpr_array = []
    
    for thresold in unique_proba:
        # 2) compute new y_hat for given {thresold}
        y_hat = np.where(d_s[y_proba] >= thresold, 1, 0)
        df = pd.DataFrame({'y': d_s[y_act], 'y_hat': y_hat})
        
        # 3) calculate tpr, fpr
        tpr, fpr = get_tpr_fpr(df)
        
        # 4) store results to resp arrays
        tpr_array.append(tpr)
        fpr_array.append(fpr)
        
    # calculate AUC score ie Area Under Curve = Integration => by Trapezoidal method
    auc_score = None
    if tpr_array and fpr_array:
        auc_score = np.trapz(tpr_array, fpr_array) 
        
    return auc_score

# 4. Accuracy Score
def calc_accuracy_score(df):
    '''
    :param d: dataframe with 2 series ie actual & predicted
    :return : accuracy score

    Formula = (TP + TN) / (FP + TP + FN + TN)
    '''
    cm = calc_binary_confusion_matrix(df)
    total = cm.sum()
    true_pred_total = np.diag(cm).sum() 
    accuracy = true_pred_total/total
    return accuracy

def mean_sq_err(d):
    '''
    :param d: dataframe with 2 series ie actual & predicted
    :return : Mean Squared error

    Formula = 1/n * Sigma[(y - y`)^2]
    '''
    y, y_hat = d.columns
    n = d.shape[0]
    return pow(d[y] - d[y_hat], 2).sum() / n


def mape(d):
    '''
    Calculate MAPE (Mean Absolute Percentage Error) for given dataset d with 2 Series ie actual & predicted
    '''
    y, y_hat = d.columns
    err = (d[y_hat] - d[y]).abs() 
    return sum(err) / sum(d[y].abs())


def r_square(d):
    '''
    calcuklate R-Squared error for dataset d with 2 seires ie actual & predicted
    '''
    y, y_hat = d.columns
    mean = d[y].mean()
    
    ss_total = sum((d[y_hat] - mean)**2)
    ss_residue = sum((d[y_hat] - d[y])**2)
    
    return 1 - ss_residue/ss_total