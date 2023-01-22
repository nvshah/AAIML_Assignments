
def curve_smoothing(string):
    '''
    Goal :- Divide the left & right val to in between _
    Idea ;- left & right pointer with right as leading to left
    Time :- complexity O(n^2)
    '''
    # get each character by removing the commas
    s = string.split(',')
    size = len(s)

    if size <= 1: # edge case
        return string

    lp = 0   # left Pointer
    # Check At Start (Edge Case 1 ie '_' or num)
    if s[0] != '_': # decide left value
        lv = s[0] = int(s[0])
    else:
        lv = 0 

    for i, c in enumerate(s[1:], 1):
        if c != '_' and i != lp+1:  # fill space required
            r = int(c)  # right value
            cnt = i - lp + 1 # no. of character needs to be altered
            v = (r + lv) // cnt  # value needs to be fill at @cnt places
            s[lp:i+1] = [v]*cnt # alteration
            lp, lv = i, v  # update left ptr & left val

    # Check At End (Edge Case 2 ie '_' or num)
    if lp != size-1: # if last character is '_'
        cnt = size-lp 
        v = s[lp] // cnt 
        s[lp:] = [v]*cnt
            
    return ','.join(map(str, s))

s =  "_,_,30,_,_,_,50,_,_"
s =  "_,_,_,24"
s = "40,_,_,_,60"
s = "80,_,_,_,_"
s = "_,_,54,_,_"
smoothed_values= curve_smoothing(s)
print(smoothed_values)