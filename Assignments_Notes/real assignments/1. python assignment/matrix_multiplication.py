'''
Matrix Multiplication Applying Multiple Transformation in Row from Right to Left
'''

from typing import List
from operator import mul

def calculate_matrix_mul(A, B):
    ''' Matrix Multiplication ie Dot Product | Cross Product '''
    def _app1(A, B):
        # transpose of a matrix B to get column at a time
        *B_t, = zip(*B)
        # calculate mat-mul i.e row_of_A * col_of_B at a time
        return [[sum(map(mul, r_a, c_b)) for c_b in B_t]  for r_a in A]

    def _app2(A, B):
        r_a, r_b, c_d = len(A), len(B), len(A[0])
        return [
            sum(A[i][k] * B[k][j] for k in range(c_d))  # row * col elem dot product
            for i in range(r_a)
            for j in range(r_b)
        ]  
    
    return _app2(A, B)      


def matrix_mul(A: List[List[int]], B: List[List[int]]):
    # write your code
    assert len(A) and len(B), "A*B = Not Possible"
    r_b, c_a = len(B), len(A[0])

    # Edge case
    if c_a != r_b:
        raise Exception("A*B = Not Possible")

    # transpose of a matrix B to get column at a time
    *B_t, = zip(*B)
    # calculate mat-mul i.e row_of_A * col_of_B at a time
    product = [[sum(map(mul, r_a, c_b)) for c_b in B_t]  for r_a in A]

    multiplication_of_A_and_B  = "A*B = {}".format(product)
    return(multiplication_of_A_and_B)

def matrix_mul2(A: List[List[int]], B: List[List[int]]):
    """ matrix multiplication of A * B  """

    # write your code
    assert len(A) and len(B), "A*B = Not Possible"
    r_a, r_b, c_a = len(A), len(B), len(A[0])

    # Edge case
    if c_a != r_b:
        raise Exception("A*B = Not Possible")
    
    c_d = r_b  # common dimen

    matmul =  [
        sum(A[i][k] * B[k][j] for k in range(c_d))  # row * col elem dot product
        for i in range(r_a)
        for j in range(r_b)
    ]

    multiplication_of_A_and_B  = "A*B = {}".format(matmul)
    return(multiplication_of_A_and_B)




A = [[1, 3, 4],
     [2, 5, 7],
     [5, 9, 6]]
B = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]

A = [[1,2], [3,4]]
B = [[1,2,3,4,5], [5,6,7,8,9]]

A = [[1,2], [3,4]]
B = [[1,4], [5,6], [7,8], [9,6]]

ans = matrix_mul(A, B)

print(ans)
