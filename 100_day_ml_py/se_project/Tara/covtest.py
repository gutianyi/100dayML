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
def read_testcase(testCaseList, filePath):
    # testCaseList = []
    with open(filePath, "r") as f:
        while True:
            test_data = f.readline()
            if not test_data:
                break
            test_data.split()
            testCaseList.append(test_data)


# 打开JSON文件并且读取里面的内容
def read_json_exe():
    with open("coverage.json", "r") as ff:
        result_read = json.load(ff)

        return result_read['files']['mid.py']['executed_lines']


def read_json_mis():
    with open("coverage.json", "r") as ff:
        result_read = json.load(ff)

        return result_read['files']['mid.py']['missing_lines']
