import coverage
import mid
# import bubble_sort
# import merge_sort
import covtest
import math
import numpy as np
import time

'''
                  [Crosstab for each statement]
|                      | ω is covered | ω is not covered | Σ   |
|successful executions |      N_cs(ω) |          N_us(ω) | N_s |
|    failed executions |      N_cf(ω) |          N_uf(ω) | N_f |
|                    Σ |       N_c(ω) |           N_u(ω) | N   |

E_cf(ω) = (N_c(ω) * N_f) / N
E_cs(ω) = (N_c(ω) * N_s) / N
E_uf(ω) = (N_u(ω) * N_f) / N
E_us(ω) = (N_u(ω) * N_s) / N
Chi_square(ω) = ( math.pow( ( N_cf(ω) - E_cf(ω) ), 2) / E_cf(ω) ) + 
                ( math.pow( ( N_cs(ω) - E_cs(ω) ), 2) / E_cs(ω) ) + 
                ( math.pow( ( N_uf(ω) - E_uf(ω) ), 2) / E_uf(ω) ) + 
                ( math.pow( ( N_us(ω) - E_us(ω) ), 2) / E_us(ω) )

In our case, row = col = 2  ==>  math.sqrt( (row - 1)*(col - 1) ) = 1
contingency_coefficient(ω) = (Chi_square(ω) / N) / math.sqrt( (row - 1)*(col - 1) )
                              = Chi_square(ω) / N

Phi(ω) = (N_cf(ω) * N_s) / (N_cs(ω) * N_f)

if Phi(ω) > 1
    Zeta(ω) = contingency_coefficient(ω)
elif Phi(ω) = 1
    Zeta(ω) = 0
elif Phi(ω) < 1
    Zeta(ω) = -contingency_coefficient(ω)

'''
start = time.time()
# 得到需要测试的代码的行数
lines_num = covtest.getcodelines("mid.py")

N_cs, N_us, N_s = [0] * lines_num, [0] * lines_num, 0
N_cf, N_uf, N_f = [0] * lines_num, [0] * lines_num, 0
N_c, N_u, N = 0, 0, 0
E_cf, E_cs, E_uf, E_us = 0, 0, 0, 0
Chi_square, contingency_coefficient, Phi, Zeta = [0] * lines_num, [0] * lines_num, [0] * lines_num, [0] * lines_num

# 读取test case
testCaseList = []  # 存test case；一行行存
testCaseResultList = []  # 存test case result

# 读取test case
covtest.read_testcase(testCaseList, "test_case(mid)-5000/test_case.txt")
covtest.read_testcase(testCaseResultList, "test_case(mid)-5000/test_case_result.txt")

# 设置coverage matrix 行数 = 待测文件行 + 1 (多一行显示P/F)；列数 = 测试用例数目
coverage_matrix = [[0 for i in range(len(testCaseList))] for j in range(lines_num)]

# 运行结果
run_result = [0 for k in range(len(testCaseList))]

for case_num in range(len(testCaseList)):
    cov = coverage.Coverage()
    cov.start()

    # test case应得结果
    test_case_result = list(map(int, testCaseResultList[case_num].split(',')))
    # test case
    test_array = list(map(int, testCaseList[case_num].split(',')))

    # 运行待测程序得到运行结果
    result = mid.mid(test_array[0], test_array[1], test_array[2])
    #result = bubble_sort.bubble_sort(test_array)

    # 比较运行结果与应得结果，相等则P否则F
    if result == test_case_result[0]:
        isPass = 0
    else:
        isPass = 1

    # 记录test case的P/F情况
    run_result[case_num] = isPass

    cov.stop()
    cov.save()
    cov.json_report()

    # 覆盖/未覆盖的语句
    statement_exe = covtest.read_json_exe()
    statement_mis = covtest.read_json_mis()

    # 读取json文件中覆盖到的语句序号，并且在cov matrix中把对应的赋值为1
    for j in range(len(statement_exe)):
        coverage_matrix[statement_exe[j] - 1][case_num] = 1

        if isPass == 0:
            N_cs[statement_exe[j] - 1] += 1
        elif isPass == 1:
            N_cf[statement_exe[j] - 1] += 1
    for k in range(len(statement_mis)):
        if isPass == 0:
            N_us[statement_mis[k] - 1] += 1
        elif isPass == 1:
            N_uf[statement_mis[k] - 1] += 1

# 计算并得到Statistics for statements
for lineNum in range(lines_num):
    N_s, N_f = N_cs[lineNum] + N_us[lineNum], N_cf[lineNum] + N_uf[lineNum]
    N_c, N_u, N = N_cs[lineNum] + N_cf[lineNum], N_us[lineNum] + N_uf[lineNum], N_s + N_f
    E_cs = (N_c * N_s) / N
    E_us = (N_u * N_s) / N
    E_cf = (N_c * N_f) / N
    E_uf = (N_u * N_f) / N

    if E_cs != 0 and E_us != 0 and E_cf != 0 and E_uf != 0:
        Chi_square[lineNum] = (math.pow((N_cf[lineNum] - E_cf), 2) / E_cf) + (
                math.pow((N_cs[lineNum] - E_cs), 2) / E_cs) + (
                                      math.pow((N_uf[lineNum] - E_uf), 2) / E_uf) + (
                                      math.pow((N_us[lineNum] - E_us), 2) / E_us)

    contingency_coefficient[lineNum] = Chi_square[lineNum] / N

    if (N_cs[lineNum] * N_f) != 0:
        Phi[lineNum] = (N_cf[lineNum] * N_s) / (N_cs[lineNum] * N_f)
    elif (N_cs[lineNum] * N_f) == 0:
        Phi[lineNum] = 2  # 除以0表示正无穷，这里用2代表正无穷

    if Phi[lineNum] > 1:
        Zeta[lineNum] = contingency_coefficient[lineNum]
    elif Phi[lineNum] == 1:
        Zeta[lineNum] = 0
    elif Phi[lineNum] < 1:
        Zeta[lineNum] = -contingency_coefficient[lineNum]

# print("Chi_square: ", Chi_square)
# print("contingency_coefficient: ", contingency_coefficient)
# print("Phi: ", Phi)
print("Zeta: ", Zeta)

print("The Buggy statement is Line: ", Zeta.index(max(Zeta)) + 1)
arr = np.array(Zeta)
print(np.argsort(-arr) + 1)

end = time.time()
print('time cost: ',end - start)
