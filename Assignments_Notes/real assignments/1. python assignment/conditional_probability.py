# write your python code here
# you can take the above example as sample input for your program to test
# it should work for any general input try not to hard code for only given input strings

from collections import defaultdict
import operator as op 
from fractions import Fraction

# you can free to change all these codes/structure
def compute_conditional_probabilites(A):
    '''
        P(A|B) = P(A.intersect(B)) / P(B)
    '''
    
    dm = defaultdict(lambda : defaultdict(int))  # data matrix for f.intersect(b) freq cnt
    n = len(A) # rows

    freq_s = defaultdict(int) # freq cnt for second col

    s1 = set() # unique vals

    for r, c in A:  # compute necessary probabilities
        freq_s[c] += 1
        dm[r][c] += 1
        s1.add(r)

    for i in s1:
        for j in freq_s.keys():
            #ans = Fraction(dm[i][j], freq_s[j]) if dm[i][j] and freq_s[j] else f'{dm[i][j]}/{freq_s[j]}'
            if dm[i][j] and freq_s[j]: 
                ans = Fraction(dm[i][j], freq_s[j]) 
            else:
                ans = f'{dm[i][j]}/{freq_s[j]}'
                
            print(f'P(F={i}|S=={j})={ans}', end= ', ')
        print()
    
A = [['F1','S1'],['F2','S2'],['F3','S3'],['F1','S2'],['F2','S3'],['F3','S2'],['F2','S1'],['F4','S1'],['F4','S3'],['F5','S1']]

compute_conditional_probabilites(A)