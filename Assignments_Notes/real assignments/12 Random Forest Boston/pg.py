#pg

import random

k = random.sample(range(20), 8)
print(k)
n = random.choices(k, k=5)
print(n)
