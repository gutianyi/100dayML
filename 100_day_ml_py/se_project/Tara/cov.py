import coverage
import mid
import covtest

totalPassed = 0
totalFailed = 0

# 得到需要测试的代码的行数
lines_num = covtest.getcodelines("mid.py")

# 读取test case
testCaseList = []  # 存test case；一行行存
covtest.read_testcase(testCaseList)

# 设置coverage matrix 行数 = 待测文件行 + 1 (多一行显示P/F)；列数 = 测试用例数目
coverage_matrix = [[0 for i in range(len(testCaseList))] for j in range(lines_num + 1)]
# print(coverage_matrix)

for case_num in range(len(testCaseList)):
    cov = coverage.Coverage()
    # print("i: ", i)
    cov.start()

    test_array = testCaseList[case_num].split(',')

    result = mid.mid(test_array[0], test_array[1], test_array[2])
    # print("test: ", int(test_array[3]))
    # print("result: ", result)
    # print("result == test_array[3]: ", int(result) == int(test_array[3]))
    if int(result) == int(test_array[3]):
        isPass = 0
        totalPassed += 1
    else:
        isPass = 1
        totalFailed += 1
    # print("isPass: ", isPass)
    coverage_matrix[-1][case_num] = isPass
    # print(coverage_matrix)
    cov.stop()
    cov.save()
    cov.json_report()

    # 覆盖/未覆盖的语句
    statement_exe = covtest.read_json_exe()
    print("statement_exe: ", statement_exe)
    # statement_mis = covtest.read_json_mis()
    for j in range(len(statement_exe)):
        # print(statement_exe[j] - 1)
        coverage_matrix[statement_exe[j] - 1][case_num] = 1
    print(coverage_matrix)
    print("totalPassed: ", totalPassed)
    print("totalFailed: ", totalFailed)


#### tarantula部分


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

print("state_P: ", state_P)
print("state_F: ", state_F)
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
# outfile is the path to write the file to, “-” will write to stdout.
# cov.xml_report()
# print(cov.json_report())
