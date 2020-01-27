# This file is a micro lib for helping create coverage matrix and result vector as the training data set and label
import json


# get number of line of testing file
def get_codelines(tested_program_filePath):
    lines = 0
    with open(tested_program_filePath, 'r') as fp:
        while True:
            count = fp.readline()
            if not count:
                break
            lines += 1
    # lines = lines - 2  # ignore the first line (def function()), and last line (return ***)
    return lines


# get test case from .txt and save test case into list (line by line)
# or used to get test case result from .txt and save test case result into list (line by line)
def get_testcase(filePath):
    test_case_list = []
    f = open(filePath, 'r')
    while True:
        test_data = f.readline()
        if not test_data:
            break
        test_data = test_data.replace('\n', '')
        test_case_list.append(test_data)
    f.close()
    return test_case_list


# open .json file and read the content of .json
def read_json_exe(tested_program_fileName):
    with open("coverage.json", "r") as ff:
        result_read = json.load(ff)
        return result_read['files'][tested_program_fileName]['executed_lines']