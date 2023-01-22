import re

def replace_digits(s):
    pattern = r"[^\d]"
    repl = ""
    result = re.sub(pattern, repl, s, 0)
    return '#'*len(result)

a = "234"
a = "a2b3c4"
a = "abv"
a = '#2a$#b%c%561#'
print(replace_digits(a))