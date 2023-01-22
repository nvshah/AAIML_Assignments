from utils import get_digits
from functools import reduce
from operator import mul
from math import prod

def prodDigits(n):
    g = get_digits(n)
    #return reduce(mul, g, 1)
    return prod(g)

print(prodDigits(22))

def MDR(n):
    '''
    all digits of num multiply with each other & repeating phenomennon till 1 digit num is obtained
    that 1 digit num is called as MDR
    & 
    num of times you did phenomenon is called as MPersistence
    86 -> 48 -> 32 -> 6 (MDR = 6, MPersistence = 3)
    341 -> 12 -> 2 (MDR = 2, MPersistence = 2)

    :return : (MDR, MPersistence)
    '''
    m_persistence = 0
    while (n // 10) != 0:
        n = prodDigits(n)
        m_persistence += 1

    return n, m_persistence

print(MDR(341))
