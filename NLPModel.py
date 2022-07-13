import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __int__(self, input_size, hidden_size, num_Classes): # feed forward neural network with 2 hidden layers
        # we will have BOG as input and # different patters and hidden layers and # of classes then we will find the SoftMax and get probability of all the classes
        super(NeuralNetwork, self).__int__() # to call the self
        self.layer1=nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_Classes)
        #hidden size can be changed but number of classes remains the same
        #activation function
        self.relu=nn.ReLU()

    def forward(self,n):
        net=self.layer1(n)
        net = self.relu(net)# activation function
        net = self.layer2(net)
        net=self.relu(net) # activation function to activate the network for the future nodes
        net = self.layer3(net)
        # we dont  do the activation function here and no softmax
        return net