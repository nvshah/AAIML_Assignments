from typing import List


def permutation(l: List[int]):
    '''
    Find Permutation using Recurssion
    :param: l -> list
    :return: -> all permutations i.e 
    p(n, r) = n! / (n-r)!
    '''
    # Base Case -> At Most 1 element
    if len(l) <= 1:
        return [l]

    ans = []

    for e in l:
        fix_e = [i for i in l if e != i]
        perm_rest = permutation(fix_e)
        ans.extend(([e, *p] for p in perm_rest))

    return ans

def fact(n):
    ''' calculating factorial using non-tailed recursion '''
    if n == 1:
        return 1
    if n == 2:
        return 2
    return n * fact(n-1)

def permutation(n, r):
    '''
        p(n, r) = n! / (n-r)!
    '''
    return fact(n) / fact(n-r)

def combinations(n, r):
    '''
        c(n, r) = n! / (r! * (n-r)!) 
                = p(n,r) / r!
    '''
    return permutation(n, r) / fact(r)

def combination(l, r):
    '''
    Find Combination using Recurrsion
     :param : l -> list
     :r : r -> num of elements to pick
    '''
    assert len(l) >= r, "Length of l must be >= r" 
    
    # base Case i.e when to select = 1 => select all
    if r == 1:
        return [[e] for e in l]
    if len(l) == r:
        return [l]

    ans = []
    
    for i in range(len(l)-r+1):
        rest_comb = combination(l[i+1:], r-1)
        ans.extend([[l[i], *c] for c in rest_comb])

    return ans 