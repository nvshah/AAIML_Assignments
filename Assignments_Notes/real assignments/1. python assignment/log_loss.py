from math import log10

def compute_log_loss(A):
    '''
        Formula := -1/n * sum(yi * log10(pi) + (1-yi) * log10(1-pi))
    '''
    if not A:
        return 0
    n = len(A)
    loss = (-1/n)*sum(((y * log10(p)) + ((1-y) * log10(1-p)) for y, p in A))
    return round(loss, 5)

A = [[1, 0.4], [0, 0.5], [0, 0.9], [0, 0.3], [0, 0.6], [1, 0.1], [1, 0.9], [1, 0.8]]
loss = compute_log_loss(A)
print(loss)