import coverage
import mid
# import bubble_sort
# import merge_sort
import covtest
import numpy as np
import time

totalPassed = 0
totalFailed = 0

start = time.time()
# 得到需要测试的代码的行数
lines_num = covtest.getcodelines("mid.py")

# 读取test case
testCaseList = []  # 存test case；一行行存
testCaseResultList = []  # 存test case result

covtest.read_testcase(testCaseList, "test_case(mid)-5000/test_case.txt")
covtest.read_testcase(testCaseResultList, "test_case(mid)-5000/test_case_result.txt")

# 设置coverage matrix 行数 = 待测文件行 + 1 (多一行显示P/F)；列数 = 测试用例数目
coverage_matrix = [[0 for i in range(len(testCaseList))] for j in range(lines_num + 1)]

for case_num in range(len(testCaseList)):
    cov = coverage.Coverage()

    cov.start()
    test_case_result = list(map(int, testCaseResultList[case_num].split(',')))

    test_array = list(map(int, testCaseList[case_num].split(',')))

    result = mid.mid(test_array[0], test_array[1], test_array[2])
    # result = bubble_sort.bubble_sort(test_array)
    if result == test_case_result[0]:
        isPass = 0
        totalPassed += 1
    else:
        isPass = 1
        totalFailed += 1

    coverage_matrix[-1][case_num] = isPass

    cov.stop()
    cov.save()
    cov.json_report()

    # 覆盖/未覆盖的语句
    statement_exe = covtest.read_json_exe()

    for j in range(len(statement_exe)):
        coverage_matrix[statement_exe[j] - 1][case_num] = 1

# tarantula部分

state_P = [0 for i in range(lines_num)]
state_F = [0 for i in range(lines_num)]
for mm in range(lines_num):
    failed_cov_statement = 0
    passed_cov_statement = 0
    # 得到test case的PF情况
    for k in range(len(testCaseList)):
        if coverage_matrix[-1][k] == 1:
            state_F[mm] += coverage_matrix[mm][k]
        elif coverage_matrix[-1][k] == 0:
            state_P[mm] += coverage_matrix[mm][k]

state_sus = [0 for i in range(lines_num)]
for mmm in range(lines_num):
    t_div_tf = state_F[mmm] / totalFailed
    p_div_tp = state_P[mmm] / totalPassed
    # 有时分母会为0，故设此条件
    if p_div_tp + t_div_tf == 0:
        state_sus[mmm] = 0
    else:
        state_sus[mmm] = t_div_tf / (p_div_tp + t_div_tf)
print("state_sus: ", state_sus)
print("The Buggy statement is Line: ", state_sus.index(max(state_sus)) + 1)
arr = np.array(state_sus)
print(np.argsort(-arr) + 1)

end = time.time()
print('time cost: ',end - start)