# This function is used to find the middle number from three number
# This program also use to generate test case randomly and automatically
# The corresponding result of each test case is also be generated
import numpy as np
import random


def mid(x, y, z):
    m = z
    if y < z:
        if x < y:
            m = y
        elif x < z:
            m = x
    elif x > y:
        m = y
    elif x > z:
        m = x
    return m

# open a .txt file, which use to save the random generated test case
f_testcase = open('test_case.txt', 'w')

# open a .txt file, which use to save the result of each random generated test case
f_result = open('test_case_result.txt', 'w')

# generate 100 test case for sorting program
# obtain the result to those test case by executing the bubble sort program
num_of_testcase = 5000
for i in range(num_of_testcase):
    arr = []
    for z in range(3):
        arr.append(random.randint(0, 100))

    print(arr)

    # write a random generated test case into the .txt file
    # each integer split by ','
    # each test case split by '\n' (each line has a test case)
    for j in range(len(arr)):
        f_testcase.write('%s' % arr[j])
        if j + 1 < len(arr):
            f_testcase.write(',')
        if j + 1 == len(arr):
            f_testcase.write('\n')
    print("%sth test case: %s" % (i + 1, arr))

    # call bubble sort function to order the shuffling array from large to small
    middle_number = mid(arr[0], arr[1], arr[2])

    # write the result of each random generated test case into the .txt file
    f_result.write('%s' % middle_number)
    f_result.write('\n')
    print("%sth test case result: %s" % (i+1, middle_number))
    print('\n')

# close the file stream
f_testcase.close()
f_result.close()