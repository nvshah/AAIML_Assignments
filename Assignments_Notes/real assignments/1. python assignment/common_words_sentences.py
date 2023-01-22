import re

def string_features(S1, S2):
    ''' find common & disjoint words '''
    pattern = r'\s+'
    re_c = re.compile(pattern)

    s1_words = set(re_c.split(S1))
    s2_words = set(re_c.split(S2))
    common = s1_words.intersection(s2_words)

    a = len(common)
    b = s1_words.difference(common)
    c = s2_words.difference(common) 

    return a, b, c

S1= "the first column F will contain only 5 uniques values"
S2= "the second column S will contain only 3 uniques values"
a,b,c = string_features(S1, S2)
print(a, b, c)