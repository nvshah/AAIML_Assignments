import math
import re
from itertools import zip_longest
from operator import mul
import itertools as it
# write your python code here
# you can take the above example as sample input for your program to test
# it should work for any general input try not to hard code for only given input strings

'''
Idea :- apply the eqn of line to 
         1) All red points & beware of the positions all red points must have same sign 
            despite magnitude
         2) Same goes for Blue Points ie Despite magnitude all blue pts must have same sign
            to ennsure that they belong to same partition

'''


def extract_weights(s: str):
    ''' return the coefficients of eqn of form ax+by+c '''
    # TODO make pattern for nD general
    pattern = r"(.+)x(.+)y(.+)"  # for 2D only
    m = re.match(pattern, s)
    return list(map(eval, [m.group(1), m.group(2), m.group(3)])) if m else None

def extract_coeff(s: str):
    pattern = r'[a-zA-Z]+'
    s = re.sub(pattern, " ", s)
    return list(map(eval, s.split()))

def fit_to_line_sign(pt, eqn):
    ''' Apply the eqn to pt & return the sign of magnitude '''
    v = sum(it.starmap(mul, zip_longest(eqn, pt, fillvalue=1)))
    return math.copysign(1, v)

# you can free to change all these codes/structure
def i_am_the_one(red,blue,line):
    # extract the coefficients from line equation [a, b, c]
    w = extract_weights(line)

    # get only signs of eqn fit for all pts belonging to red
    red_signs = map(lambda r: fit_to_line_sign(r, w), red)
    sign = 0
    for i in red_signs:
        if not sign and i:
            sign = i
        elif not (i + sign): # conflicts in red grp
            return "NO"
    
    # get only signs of eqn fit for all pts belonging to blue
    blue_signs = map(lambda b: fit_to_line_sign(b, w), blue)
    sign = 0
    for i in blue_signs:
        if not sign and i:
            sign = i
        elif not (i + sign): # conflicts in red grp
            return "NO"

    return "YES"

Red= [(1,1),(2,1),(4,2),(2,4), (-1,4)]
Blue= [(-2,-1),(-1,-2),(-3,-2),(-3,-1),(1,-3)]
Lines=["1x+1y+0","1x-1y+0","1x+0y-3","0x+1y-0.5"]

for i in Lines:
    yes_or_no = i_am_the_one(Red, Blue, i)
    print(yes_or_no) # the returned value

