import math
from collections import defaultdict


def prime_factors(n):
    if n <= 1:
        return None

    d = defaultdict(int)
    #s = 2
    #b = n // 2
    
    # for all even numbers
    while n % 2 == 0:
        d[2] += 1
        n = n // 2
    
    # for all odd numbers
    i = 3
    while n != 1:  # till we dont get entire number broken to single unit, divide it
        while n % i == 0:
            d[i] += 1
            n = n // i 
        i += 2

    return d
    

print(prime_factors(78))
print(prime_factors(63))