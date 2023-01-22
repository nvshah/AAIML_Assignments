l = [[1,2,3], [4,5,6], [7,8,9]]

for i in range(len(l)):
    g = [n for g in range(len(l)) if g != i for n in l[g]]
    print(g)