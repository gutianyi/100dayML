import copy
import json


# 获取待测试文件的行数
def getcodelines(filePath):
    lines = 0
    with open(filePath, "r") as fp:
        while True:
            count = fp.readline()
            if not count:
                break
            lines += 1
    return lines


# 从.txt文件里读取测试用例并且将其存入list中(整行存，没split)
def read_testcase(testCaseList):
    # testCaseList = []
    with open("testCase.txt", "r") as f:
        while True:
            test_data = f.readline()
            if not test_data:
                break
            test_data.split()
            testCaseList.append(test_data)
            # print(test_data)
            # test_array = test_data.split(',')

        # print(testCaseList[0])
        # print(len(testCaseList))


# 打开JSON文件并且读取里面的内容
def read_json_exe():
    with open("coverage.json", "r") as ff:
        result_read = json.load(ff)
        # print(result_read)
        # statement_num = result_read['files']['mid.py']['summary']['num_statements']
        # statement_exe.append(result_read['files']['mid.py']['executed_lines'])

        return result_read['files']['mid.py']['executed_lines']


def read_json_mis():
    with open("coverage.json", "r") as ff:
        result_read = json.load(ff)
        # print(result_read)
        # statement_num = result_read['files']['mid.py']['summary']['num_statements']
        # statement_exe.append(result_read['files']['mid.py']['executed_lines'])
        return result_read['files']['mid.py']['missing_lines']
