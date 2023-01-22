from collections import Counter
from itertools import accumulate

from random import random, choices

def ceil(l, target):
    '''
        Compute the interval upper-bound for target via binary search
    '''
    s = len(l)
    start, end = 0, s - 1
    while start <= end:
        mid = start + ((end - start) // 2)
        if target == l[mid]:
            return mid
        if target > l[mid]:
            start = mid + 1
        else:
            end = mid - 1
    return start

def pick_a_number_from_list(lst):
    '''
        Proportional Sampling Case
    '''
    # step1 - sum of all numbers
    s = sum(lst)
    # step2 - normalize all numbers
    n_lst = [e/s for e in lst]
    # step3 - accumulate
    *cum_norm_sum, = accumulate(n_lst)
    # step4 - find proper ceil idx
    idx = ceil(cum_norm_sum, random())
    selected_random_number = lst[idx]

    return selected_random_number

def sampling_based_on_magnitued(l, ntimes = 100):
    '''
        Proportional sampling via theoretical formulation via cumulative sums & ratios
    '''
    d = dict.fromkeys(l, 0) # to keep track of freq of each member during testing
    r = ntimes  # times to perform test
    for i in range(1,r+1): # test
        number = pick_a_number_from_list(A)
        d[number] += 1
    print('Probabilities :')

    pairs = sorted(d.items(), key=lambda x: x[1], reverse=True)
    for pair in pairs:
        print(pair[0], '-> ', format(pair[1]*100/r, '.2f'), '%')

def sampling_based_on_magnitued_a2(l, ntimes=100):
    ''' 
        Proportional Sampling via random.choices()
    '''
    total= sum(l)
    k = ntimes 
    perctg = [(k*n)/total for n in l] # k times (% of single time)
    r = choices(l, weights=perctg, k=k) 
    ctr = Counter(list(r))

    pairs = sorted(ctr.items(), key=lambda x:x[1], reverse=True)
    for pair in pairs:
        print(pair[0], '-> ', format(pair[1]*100/ntimes, '.2f'), '%')

#sampling_based_on_magnitued()

if __name__ == '__main__':
    # l = [1, 2, 4, 10, 12]
    # ans = ceil(l, 10.1)
    # print(ans)

    A = [0, 5, 27, 6, 13, 28, 100, 45, 10, 79]
    sampling_based_on_magnitued(A, 100000)
    print('---------')
    sampling_based_on_magnitued_a2(A, 100000)