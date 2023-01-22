from collections import deque
from utils import get_digits

def decimal_to_binary(n):
    #return bin(n)
    #return f'{n:b}'
    res = deque()

    while True:
        n, r = divmod(n, 2)
        res.appendleft(r)
        if not n:
            break

    return ''.join((map(str, res)))

def decimal_to_binary2(n):
    return ''.join(map(str, get_digits(n, base=2)))[::-1]

def decimal_to_base(n, b):
    return ''.join(map(str, get_digits(n, base=b)))[::-1]

#print(decimal_to_base(12, 6))


#print(decimal_to_binary(16))

print(reversed('dsdad'))

