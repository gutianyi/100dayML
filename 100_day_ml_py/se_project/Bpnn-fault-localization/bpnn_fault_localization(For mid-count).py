import numpy as np
import scipy.special
import coverage
import covlib
import Mid_1b_v1 as mid_1b_v1
import Mid_1b_v2 as mid_1b_v2
import Mid_1b_v3 as mid_1b_v3
import Mid_1b_v4 as mid_1b_v4
import Mid_2b_v1 as mid_2b_v1
import Mid_2b_v2 as mid_2b_v2
import Mid_2b_v3 as mid_2b_v3
import Mid_2b_v4 as mid_2b_v4
import Mid_2b_v5 as mid_2b_v5
import Mid_2b_v6 as mid_2b_v6
import Mid_3b_v1 as mid_3b_v1
import Mid_3b_v2 as mid_3b_v2
import Mid_3b_v3 as mid_3b_v3
import Mid_3b_v4 as mid_3b_v4
import Mid_4b_v1 as mid_4b_v1

# np.set_printoptions(threshold=np.inf)
# neural network class definition
class NeuralNetwork:

    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, samplesize):
        # set number of nodes in each input layer, hidden layer, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link weight matrices, wih and who are weight matrices
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # aka link weight matrix between input and hidden layer [w11 w21 ... wm1]
        # (m is number of input nodes)                          [w12 w22 ... wm2]
        #                                                       [w13 w23 ... wm3]
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate

        # obtain the number of samples for each batch_size
        # create a cache for store the output layer error of each samples of an iteration
        self.samplesize = samplesize
        self.cache_errors = np.zeros((1, self.samplesize))

        # activiation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        # index used to notice the program that N samples had already done
        self.index = 0

    # train the neural network
    def train(self, inputs_list, targets_list):
        # index + 1 when a sample input
        self.index += 1

        # convert input_list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # forward propagation the signal and obtain the outputs of each layer
        hidden_outputs, final_outputs = self.forward_propagation(inputs)

        # backward propagation the errors for each layer (for output layer and hidden layer)
        hidden_errors, output_errors = self.back_propagation(final_outputs, targets)

        # calculate the cost function(aka MSE) when the training of a batch_size samples is finish
        self.cost_calculation(output_errors)

        # update the weight
        self.weights_update(inputs, hidden_outputs, final_outputs, hidden_errors, output_errors)

    # forward propagation signal
    def forward_propagation(self, inputs):
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return hidden_outputs, final_outputs

    # backward propagation errors
    def back_propagation(self, final_outputs, targets):
        # calculate output layer's error: (target - actual)
        output_errors = targets - final_outputs

        # calculate hidden layer's error:
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        return hidden_errors, output_errors

    def weights_update(self, inputs, hidden_outputs, final_outputs, hidden_errors, output_errors):
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

    def cost_calculation(self, output_errors):
        # cache the output layer errors of 'index'th sample
        if int(self.index) <= int(self.samplesize):
            self.cache_errors[0][self.index - 1] = output_errors
        # calculate the cost function when number of batch_size sample done (in this case, cost function is MSE)
        if int(self.index) == int(self.samplesize):
            error_sum = np.sum(self.cache_errors, axis=1)
            loss = (1 / self.samplesize) * pow(error_sum[0], 2)  # MSE
            print(loss)
            # initialise index for next iteration
            self.index = 0
            return loss

    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# This function is used to generate training set and corresponding label
def coverage_generate(tested_program_path, tested_program_name, test_case_path, test_case_result_path):
    # get number of code of the program to be tested
    statement_num_sum = covlib.get_codelines(tested_program_path)
    # get the test case (get_testcase() return a list to test_case variable)
    test_case = covlib.get_testcase(test_case_path)
    # get the test case result (get_testcase() return a list to test_case_result variable)
    test_case_result = covlib.get_testcase(test_case_result_path)

    # initialise coverage matrix and execution result vector
    # e.g. coverage matrix :     s1 s2 ... sM
    #                        t1 [1  0  ...  1]
    #                        t2 [0  1  ...  0]
    #                        .. [      ...   ]
    #                        tN [1  1  ...  0]
    # '1' implies s(x) cover by t(y); '0' implies s(x) is not cover by t(y)
    coverage_matrix = np.zeros((len(test_case), statement_num_sum))

    # e.g. result vector:  t1 t2 ... tN
    #                      [1  0 ... 1]
    # '1' implies t(y) is failed; '0' implies t(y) is success
    result_vector = np.zeros((1, len(test_case)))

    # update coverage matrix and result vector
    for num_case in range(len(test_case)):
        cov = coverage.Coverage()
        cov.start()

        # map(int, list): transform string type elements to integer type elements
        test_array = list(map(int, test_case[num_case].split(',')))
        test_result_array = list(map(int, test_case_result[num_case].split(',')))

        # input the test case(test_array) to the program to be tested
        # the actual_result may be a list or a value

        actual_result = mid_4b_v1.mid(test_array[0], test_array[1], test_array[2])  # ***调用待测程序(根据需要测试.py修改)***


        cov.stop()
        cov.save()
        # generate a coverage information report
        cov.json_report()

        # extract the statement coverage data (executed lines) from .json file
        statement_exe = covlib.read_json_exe(tested_program_name)

        # update the coverage_matrix
        for num_stat in range(len(statement_exe)):
            coverage_matrix[num_case][statement_exe[num_stat] - 1] = 1
        # update the result_vector
        for num_result in range(len(test_result_array)):
            # the actual_result may be a list or a value
            # please according to the return parameter of the function to revise the code
            if actual_result != test_result_array[num_result]:  # ***actual_result需要根据函数的返回值修改***
                result_vector[0][num_case] = 1
    return coverage_matrix, result_vector


# enter the file path
tested_program_filePath = 'Mid_4b_v1.py'  # ***待测程序的路径***
tested_program_fileName = 'Mid_4b_v1.py'  # ***待测程序的名称(包含后缀)***
test_case_filePath = 'test_case(mid)/test_case.txt'  # ***待测程序的测试用例的路径***
test_case_result_filePath = 'test_case(mid)/test_case_result.txt'  # ***待测程序的测试用例的正确输出的路径***

# training set and testing set
# coverage vector matrix (training set)
# execution result of each coverage vector (label of training set)
training_data_list, training_target_list = coverage_generate(tested_program_filePath, tested_program_fileName,
                                                             test_case_filePath, test_case_result_filePath)
training_target_list = training_target_list.T

# virtual coverage vector matrix (testing set)
testing_data_list = np.zeros((len(training_data_list[0]), len(training_data_list[0])))
# generate a M*M identity matrix
for i in range(len(testing_data_list[0])):
    testing_data_list[i][i] = 1

# suppose 'N' represent the number of coverage vector (input or sample)
# N = 7 (base ib above coverage data)
N = len(training_data_list)
# suppose 'M' represent the number of statement
# M = 9 (base on above coverage data)
M = len(training_data_list[0])
# number of input layer, hidden layer, output layer nodes, and batch_size
input_nodes = M
hidden_nodes = 3
output_nodes = 1
learning_rate = 0.5
sample_size = N
# create an object of neuralNetwork class
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate, sample_size)

import time
start = time.time()

# training the neural network
iteration = 1000
for i in range(iteration):
    print("The loss of %sth iteration: " % (i + 1))
    for record in range(sample_size):
        inputs = np.asfarray(training_data_list[record])
        targets = np.asfarray(training_target_list[record])
        # the output range of sigmoid function is (0,1)
        # if the target(label of training set) set 1.0 or 0.0, it may cause the weights of network is too big
        # which may cause neural network saturation.
        if int(targets[0]) == int(1):
            targets[0] = 0.99
        elif int(targets[0]) == int(0):
            targets[0] = 0.01
        n.train(inputs, targets)


# using this method to rank the suspiciousness with statement index
def bubble_sort(arr, i_list):
    for i in range(len(arr)-1):
        for j in range(len(arr)-1):
            if arr[j] < arr[j+1]:
                temp = arr[j+1]
                arr[j+1] = arr[j]
                arr[j] = temp

                temp = i_list[j+1]
                i_list[j+1] = i_list[j]
                i_list[j] = temp
    return arr, i_list

end = time.time()

# testing the neural network
index_list = []
suspiciousness_list = []
for record in range(len(testing_data_list)):
    inputs = np.asfarray(testing_data_list[record])
    outputs = n.query(inputs)
    suspiciousness_list.append(outputs[0][0])
    index_list.append(record+1)
    print("The suspiciousness of statement %s is %s" % (record + 1, outputs[0]))


suspiciousness_list, index_list = bubble_sort(suspiciousness_list, index_list)
for i in range(len(index_list)):
    print("Rank %s: (statement %s, suspiciousness = %s)" % (i+1, index_list[i], suspiciousness_list[i]))

print('训练花费时间',end-start,'s')