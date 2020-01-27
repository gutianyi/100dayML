def merge(arr, l, m, r):
    n1 = m - l + 1
    n2 = r - m
    L = [0] * (n1)      # 创建临时数组
    R = [0] * (n2)      # 创建临时数组
    for i in range(0, n1):
        L[i] = arr[l + i]   # 拷贝数据到临时数组 arrays L[] 和 R[]
    for j in range(0, n2):
        R[j] = arr[m + 1 + j]   # 拷贝数据到临时数组 arrays L[] 和 R[]
    i = 0  # 初始化第一个子数组的索引
    j = 0  # 初始化第二个子数组的索引
    k = l  # 初始归并子数组的索引
    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        elif L[i] > R[j]:
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
def mergeSort(arr, l, r):
    if l < r:
        m = int((l + (r - 1)) / 2)
        mergeSort(arr, l, m)
        mergeSort(arr, m + 1, r)
        merge(arr, l, m, r)