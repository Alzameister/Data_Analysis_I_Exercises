import numpy as np

# Do not do several variable names but use e.g. a list and indices

# DON'T (like never)
l1 = []
l2 = []
l3 = []

# bad DON'T do it either here, minimally better. Sometimes required if every element is different
list1 = [[], [], [], [] , [], []]

# good
list1 = []
for i in range(6):
    list1.append([])

# best (but a new concept)
list1 = [[] for _ in range(6)]  # list comprehension


# CHOOSE GOOD NAMES
# not list1 as above, say what it is
age_counts = [[] for _ in range(6)]

lower_age = 5  # as an example: this makes things more verbose, which is good!
upper_age = 82
n_runners = 1304
ages = np.random.uniform(lower_age, upper_age, size=n_runners)  # just an example to work with something

# It's okay to do it once on your own to maybe understand an algorithm. But in general, and for future reference,
# use any available function as demonstrated here

# DON'T never
for i in range(len(ages)):
    if age[i] < 20:
        list1[0].append(ages[i])
    elif 21 <= ages[i] <= 30:
        list1[1].append(ages[i])
    elif 31 <= ages[i] <= 40:
        list1[2].append(ages[i])
    elif 41 <= ages[i] <= 50:
        list1[3].append(ages[i])
    elif 51 <= ages[i] <= 60:
        list1[4].append(ages[i])
    else:
        list1[5].append(ages[i])
        
# Okay for training, just to understand the concept or if explicitly requested

for age in ages:
    if age < 20:
        age_counts[0].append(age)
    # ... and so on
    
# even better, think about it before! Sketch on pen and paper and try to implement something smart (but don't over-do...), for example
edges = [20, 50, 60, 999] # 999 is like the maximum age
lower_edge = 0
for i, upper_edge in enumerate(edges):
    for age in ages:
        if lower_edge <= age <= upper_edge:
            age_counts[i].append(age)
    lower_edge = upper edge

# DO in general if you understand the concepts and can use other functions
edges = np.array([0, 20, 30, 40, 50, 60, 100])
bincount, edges = np.histogram(age, bins=edges)

# USEFUL SYNTAX
# Array slicing
rnd1 = np.random.normal(size=(1000, 8))  # creating a random array for demonstration purpose, could be ironman.txt
# This has 1000 rows and 8 columns. To select e.g. column 3 (from 0-7), do
col3 = rnd1[:, 3]  # use a better name than "col3" if you know what it is. "Years", "swimming_times", "ranks", ...

# The general syntax here is: [element_of_first_axis, element_of_second_axis, ...]
# where the first axis is the row, second the column, and so on
# ':' is slicing. Instead of selecting a single element of an array, we can select a range
# 'start:stop' where start index is included, stop is excluded. The following selects the elements 2-5 in the row
elements_2_to_5 = rnd1[2:5]
# if start or stop is left away, it means "from the beginning" or "to the end" respectively. So ':' means "everything"
# This is what is done above
col0 = rnd1[:, 3]
# selects "everything" in the first axis (the rows) and the third element from the second axis ("columns")



# DEBUGGING and understanding the problems
# make sure that you understand what is going on! If not, look at the problem. There are two good ways:
# a) use the command line. pycharm has a Python console that you can use to enter an expression and evaluate it immediately. Use it! For example, what does "range"do? or the "enumerate"?
print(range(5))  
# one thing: if it gives something "weird", like a generator, make it a list (try that)
print(list(range(5))
# there we go! 

# b) use the debugger (as explained in the PyCharm instructions. Set a breakpoint, wait until it is there and check the variables.

