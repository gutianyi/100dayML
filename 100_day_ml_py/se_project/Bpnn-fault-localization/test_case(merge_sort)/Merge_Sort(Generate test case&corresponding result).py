# This function is a merge sort without any bug
# This program also use to generate test case randomly and automatically
# The corresponding result of each test case is also be generated
import numpy as np


class MergeSort():
    def merge(self, arr, l, m, r):
        n1 = m - l + 1
        n2 = r - m
        L = [0] * (n1)  # 创建临时数组
        R = [0] * (n2)  # 创建临时数组
        for i in range(0, n1):
            L[i] = arr[l + i]  # 拷贝数据到临时数组 arrays L[] 和 R[]
        for j in range(0, n2):
            R[j] = arr[m + 1 + j]  # 拷贝数据到临时数组 arrays L[] 和 R[]
        i = 0  # 初始化第一个子数组的索引
        j = 0  # 初始化第二个子数组的索引
        k = l  # 初始归并子数组的索引
        while i < n1 and j < n2:
            if L[i] <= R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        while i < n1:
            arr[k] = L[i]
            i += 1
            k += 1
        while j < n2:
            arr[k] = R[j]
            j += 1
            k += 1

    def mergeSort(self, arr, l, r):
        if l < r:
            m = int((l + (r - 1)) / 2)
            self.mergeSort(arr, l, m)
            self.mergeSort(arr, m + 1, r)
            self.merge(arr, l, m, r)


# 实例化
merge_sort = MergeSort()

# open a .txt file, which use to save the random generated test case
f_testcase = open('test_case.txt', 'w')

# open a .txt file, which use to save the result of each random generated test case
f_result = open('test_case_result.txt', 'w')

# generate N number of test case for sorting program
# obtain the result to those test case by executing the bubble sort program
num_of_testcase = 5000
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
    n = len(arr)
    merge_sort.mergeSort(arr, 0, n-1)

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
