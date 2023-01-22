from operator import methodcaller
import itertools as it

def filter_odd_nums(l):
    #return filter(lambda x: x & 1, l)
    return filter(methodcaller('__rand__', 1), l)

def cube_all(l):
    #return map(lambda x: x**3, l)
    return map(methodcaller('__pow__', 3), l)

def filter_even_nums(l):
    #return filter(lambda x: not(x & 1), l)
    return it.filterfalse(methodcaller('__rand__', 1), l)

# -------

def cube_even_nums(l):
    return cube_all(filter_even_nums(l))

*ans, = filter_odd_nums([1,2,3,4,5,6])
*cubes, = cube_all([1,2,3,4])
*cube_even, = cube_even_nums([1,2,3,4,5])

print(ans)
print(cubes)
print(cube_even)
