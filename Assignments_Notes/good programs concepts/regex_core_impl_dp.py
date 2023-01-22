# Given * & ? how regex works internally find if pattern matches with string or not 
''' 
p - String   // regex pattern allowed only options domain {*, ?, [A-Z0-9]}
s - string   // contains only {*, ?, [A-Z0-9]}

W.O using re module check if match or not 

'''

def isMatch(p, s):
    '''
    T.C := 2^n  (because of Case 2,
     2 options available at each recursive level)
     & assuming n = m approx

     T(n) = T(n-1) + T(n-1) + T(1)  
    '''
    if p == s:
        return True 
    
    if p == "*":
        return True 
    
    if not (p and s): # either of string is empty
        return False 

    # case 1
    if p[0] == s[0] or p[0] == '?':
        return isMatch(p[1:], s[1:])

    # case 2
    if p[0] == "*":
        return (isMatch(p, s[1:]) or isMatch(p[1:], s))
    
    # case 3
    if p[0] != s[0]:
        return False

# def isMatch2(p, s):

#     if not (p or s): # both string are empty
#         return True 

#     if not (p and s): # either of string is empty
#         return False 

#     # if p == s:
#     #     return True 
    
#     # if p == "*":
#     #     return True 

#     if p[0] != s[0]:
#         return False

#     # case 1
#     if p[0] == s[0]:
#         return isMatch(p[1:], s[1:])  
#     elif len(p) > 1:
#         # TODO: implemenbt for ?
#         # if p[1] == '?':
#         #     return isMatch(p)

    
#     if p[0] == "?":
        

#     if p[0] == "*":
#         if len(p) == 1: # \* at the end of p so matches all
#             return True 
#         else:
#             ix = 1
#             while p[ix] == '*': # find the index of character just after *
#                 ix += 1
#                 if ix == len(p): # all are * at the end of p
#                     return True
#             else: 
#                 char = p[ix]
#                 if idx := s.index(char) != -1:  # check if that char exist in string s if not then return
#                     return isMatch(p[ix:], s[idx:])
#                 else:
#                     return False

    if __name__ == '__main__':
        p = "abc?*"
        s = "abde"



