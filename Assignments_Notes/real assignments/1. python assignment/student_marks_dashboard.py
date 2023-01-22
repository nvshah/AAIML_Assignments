def ceil(l, target):
    '''
        Compute the interval upper-bound for target via binary search
    '''
    s = len(l)
    start, end = 0, s - 1
    while start <= end:
        mid = start + ((end - start) // 2)
        if target == l[mid]:
            return mid
        if target > l[mid]:
            start = mid + 1
        else: 
            end = mid - 1
    return start

def floor(l, target):
    '''
        Compute the interval upper-bound for target via binary search
    '''
    s = len(l)
    start, end = 0, s - 1
    while start <= end:
        mid = start + ((end - start) // 2)
        if target == l[mid]:
            return mid
        if target > l[mid]:
            start = mid + 1  
        else:
            end = mid - 1
            
    return end

def display_dash_board(students, marks):
    size = len(marks)
    # list of indices based on sorted marks
    marks_argi = sorted(range(size), key=marks.__getitem__)

    # write code for computing top top 5 students
    top_5_students = [students[i] for i in marks_argi[-1:-6:-1]]
    # write code for computing top least 5 students
    least_5_students = [students[i] for i in marks_argi[:5]]
    # write code for computing top least 5 students
    low, high = marks[marks_argi[0]], marks[marks_argi[-1]]
    d = high - low
    l = (0.25 * d) + low  # 25th percentile
    h = (0.75 * d) + low  # 75th percentile
    marks_s = [marks[i] for i in marks_argi]
    idx_l, idx_u = ceil(marks_s, l), floor(marks_s, h) # lower & upper bound for descend
    students_within_25_and_75 = [students[i] for i in marks_argi[idx_l:idx_u+1]]

    return top_5_students, least_5_students, students_within_25_and_75

students=['student1','student2','student3','student4','student5','student6','student7','student8','student9','student10'] 
marks = [45, 78, 12, 14, 48, 43, 47, 98, 35, 80]

top_5_students, least_5_students, students_within_25_and_75 = display_dash_board(students, marks)

print(top_5_students)
print(least_5_students)
print(students_within_25_and_75)
