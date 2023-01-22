from utils import get_digits
from operator import methodcaller

# def cubesum(n):
#     ''' Sum of Cube of individual digits '''
#     ans = 0
#     while n != 0:
#         n, r = divmod(n, 10)
#         ans += pow(r, 3)
    
#     return ans 

def cubesum(n):
    return sum(get_digits(n, lambda x: x**3))

def isArmStrongNum(n):
    return n == cubesum(n)

print(cubesum(13))

print(isArmStrongNum(12))