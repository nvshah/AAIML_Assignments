from math import isqrt
from itertools import chain

def proper_divisors(n):
    e = isqrt(n)  # go till the sqrt(n)
    ans = [1]
    for i in range(2, e): 
        q, r = divmod(n, i)
        if r == 0:
            ans.extend([i, q])
    
    if e*e == n: # check for sqrt number ie e*e == n
        ans.append(e)
    return sorted(ans)

def sumPdivisors(n):
    return sum(proper_divisors(n))

def is_perfect_num(n):
    return n == sumPdivisors(n)

def is_amicable_pair(n1, n2):
    return sumPdivisors(n1) == n2 and sumPdivisors(n2) == n1 

print(proper_divisors(36))
print(proper_divisors(220))
print(proper_divisors(284))
