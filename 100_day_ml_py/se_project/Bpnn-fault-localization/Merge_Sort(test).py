import coverage
import Merge_Sort_0b as ms_0b
import Merge_Sort_1b as ms_1b

arr = [12, 11, 13, 5, 6, 7]
n = len(arr)
print("给定的数组", arr)

cov = coverage.Coverage()
cov.start()
ms_1b.mergeSort(arr, 0, n - 1)
cov.stop()
cov.save()
cov.json_report()

print("\n排序后的数组", arr)
