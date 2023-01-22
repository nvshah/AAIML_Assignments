import math

def isPrime(n):
    # Square Root Range
    for i in range(2, math.isqrt(n) + 1):
        if n % i == 0:
            return False 
    return True

def twin_prime(n):
    '''2 consecutive odd prime num '''
    ans = [2, 3]
    last_prime = -1
    for i in range(5, n, 2):
        if isPrime(i):
            if last_prime != -1:   # last prime needs pair
                ans.extend((last_prime, i))
                last_prime = -1
            elif ans[-1] == i-2:  # last prime exists in list
                ans.append(i)
                last_prime = -1
            else:
                # this can be first number & required pair further
                last_prime = i
        else:
            last_prime = -1

    print(ans)

if __name__ == '__main__':
    twin_prime(100)
            