# This version are used to do the experiment (e.g. bubble sort, value operation, and so forth)
import numpy as np
import scipy.special
import coverage
import covlib


# neural network class definition
class NeuralNetwork:

    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, batch_size):
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
        self.samplesize = batch_size
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
        hidden_errors, output_errors = self.back_propagation(hidden_outputs, final_outputs, targets)

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
        # calculate the signals emerging from final outpu layer
        final_outputs = self.activation_function(final_inputs)

        return hidden_outputs, final_outputs

    # backward propagation errors
    def back_propagation(self, hidden_outputs, final_outputs, targets):
        # calculate output layer's error: (target - actual)
        output_errors = targets - final_outputs

        # calculate hidden layer's error:
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        return hidden_errors, output_errors

    def weights_update(self, inputs, hidden_outputs, final_outputs, hidden_errors, output_errors):
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
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


# training set and testing set
# (example) coverage vector matrix (training set)
training_data_list = np.array([
    [1, 1, 1, 1, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0],
    [1, 1, 0, 0, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 1, 0, 1, 1],
])

# (example) execution result of each coverage vector (label of training set)
training_target_list = np.array([[0, 0, 0, 0, 0, 1, 1]]).T

print(training_target_list)
# (example) virtual coverage vector matrix (testing set)
testing_data_list = np.zeros((len(training_data_list[0]), len(training_data_list[0])))
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
learning_rate = 0.1
batch_size = N
# create an object of neuralNetwork class
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate, batch_size)


# training the neural network
iteration = 5000
for i in range(iteration):
    print("The loss of %sth iteration: " % (i+1))
    for record in range(batch_size):
        inputs = np.asfarray(training_data_list[record])
        targets = np.asfarray(training_target_list[record])
        # the output range of sigmoid function is (0,1)
        # if the target(label of training set) is set as 1.0 or 0.0, it may cause the weights of network is too big
        # which may cause neural network saturation.
        if int(targets[0]) == int(1):
            targets[0] = 0.99
        elif int(targets[0]) == int(0):
            targets[0] = 0.01
        n.train(inputs, targets)


# testing the neural network
l = np.asfarray(np.arange(len(testing_data_list)))
for record in range(len(testing_data_list)):
    inputs = np.asfarray(testing_data_list[record])
    outputs = n.query(inputs)
    l[record] = outputs
    print("The suspiciousness of statement %s is %s" %(record+1, outputs[0]))