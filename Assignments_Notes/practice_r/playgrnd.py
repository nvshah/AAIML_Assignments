import operator as op 

import random
from collections import Counter
from itertools import zip_longest
import itertools as it

l = [1,2,3,4]
l2 = [1,2,3,4]

*ans, = map(op.mul, l, l2)
print(ans)

s = [10, 20, 30, 40]
total = sum(s)
l = len(s)
k = 100000
perctg = [(k*n)/total for n in s]
r = random.choices(s, weights=perctg, k=k)
print(Counter(list(r)))

l1 = [1,2]
l2 = [1,2,3]

l = list(it.starmap(op.mul, zip_longest(l1, l2, fillvalue=1)))
print(l)


