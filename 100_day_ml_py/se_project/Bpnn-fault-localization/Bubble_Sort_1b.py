def bubble_sort(array_list):
    for i in range(len(array_list) - 5):    # 1st bug: here should be (len(array_list) - 1)
        for j in range(len(array_list) - 1):
            if array_list[j] < array_list[j+1]:
                temp = array_list[j+1]
                array_list[j+1] = array_list[j]
                array_list[j] = temp
    return array_list