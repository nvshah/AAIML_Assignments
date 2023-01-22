import math
import itertools

# write your python code here
# you can take the above example as sample input for your program to test
# it should work for any general input try not to hard code for only given input examples
# you can free to change all these codes/structure

def calc_cos_sim(x,y):
    ''' 
    calculate the cosine similarity (ie in terms of anngle diff) between 2 vectors x & y 
    NOTE : return metric is in radian & not angle for simplicity
    '''
    n = x[0]*y[0] + x[1]*y[1]
    d = math.sqrt((x[0]**2 + x[1]**2) * (y[0]**2 + y[1]**2))
    cos_sim = math.acos(n/d)
    return cos_sim

# here S is list of tuples and P is a tuple ot len=2
def closest_points_to_p(S, P):
    cos_dist = [round(calc_cos_sim(X,P),2) for X in S] # cos-sim between p & all X
    # get first closest point to P
    closest_points_to_p = itertools.islice(map(S.__getitem__, sorted(range(len(S)), key=cos_dist.__getitem__)),5)
    return closest_points_to_p  # its list of tuples

S= [(1,2),(3,4),(-1,1),(6,-7),(0, 6),(-5,-8),(-1,-1),(6,0),(1,-1)]
P= (3,-4)
points = closest_points_to_p(S, P)
print(list(points))