# This function is a bubble sort without any bug
# This program also use to generate test case randomly and automatically
# The corresponding result of each test case is also be generated
import numpy as np


# bubble sort function, sorting the number from large to small
def bubble_sort(array_list):
    for i in range(len(array_list) - 1):
        for j in range(len(array_list) - 1):
            if array_list[j] < array_list[j + 1]:
                temp = array_list[j + 1]
                array_list[j + 1] = array_list[j]
                array_list[j] = temp
    return array_list


# open a .txt file, which use to save the random generated test case
f_testcase = open('test_case.txt', 'w')

# open a .txt file, which use to save the result of each random generated test case
f_result = open('test_case_result.txt', 'w')

# generate N number of test case for sorting program
# obtain the result to those test case by executing the bubble sort program
num_of_testcase = 100
for i in range(num_of_testcase):
    # generate a integer between 3 to 50
    index = np.random.randint(3, 50)
    # generate an array with index length
    arr = np.arange(index)
    # shuffle the array
    np.random.shuffle(arr)

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
    bubble_sort(arr)

    # write the result of each random generated test case into the .txt file
    # each integer split by ','
    # each result split by '\n' (each line has a result)
    for j in range(len(arr)):
        f_result.write('%s' % arr[j])
        if j + 1 < len(arr):
            f_result.write(',')
        if j + 1 == len(arr):
            f_result.write('\n')
    print("%sth test case result: %s" % (i + 1, arr))
    print('\n')

# close the file stream
f_testcase.close()
f_result.close()
