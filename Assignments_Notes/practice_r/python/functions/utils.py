# def get_digits(n, func=None):
#     '''
#     return generator of digits
#     if func provided then digit is mapped to that func & then returned
#     '''
#     while n != 0:
#         n, r = divmod(n, 10)
#         yield func(r) if func else r

def get_digits(n, func=None, base=10):
    '''
    :param n: number in decimal base 10
    :param func: func to apply to each extracted digit based on {base}
    :param base: base that needs to be considered whilst extracting digits
    return generator of digits
    if func provided then digit is mapped to that func & then returned
    '''
    while n:
        n, r = divmod(n, base)
        yield func(r) if func else r