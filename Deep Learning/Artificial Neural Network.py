
# coding: utf-8

# In[6]:


# Import our dependencies
from numpy import exp, array, random, dot, ones_like, where, log10

# Create our Artificial Neural Network class
class ArtificialNeuralNetwork():
    
    # initializing the class
    def __init__(self):
        
        # generating the same synaptic weights every time the program runs
        random.seed(1)
        
        # synaptic weights (3 × 4 Matrix) of the hidden layer 
        self.w_ij = 2 * random.rand(3, 4) - 1
        
        # synaptic weights (4 × 1 Matrix) of the output layer
        self.w_jk = 2 * random.rand(4, 1) - 1
    
    def Sigmoid(self, x):
        
        # The Sigmoid activation function will turn every input value into probabilities between 0 and 1
        # the probabilistic values help us assert which class x belongs to
        
        return 1 / (1 + exp(-x))
    
    def SigmoidDerivative(self, x):
        
        # The derivative of Sigmoid will be used to calculate the gradient during the backpropagation process
        # and help optimize the random starting synaptic weights
        
        return x * (1 - x)
    
    def crossentropyerror(self, a, y):
        
        # The cross entropy loss function
        # we use it to evaluate the performance of our model
        
        return - sum(y * log10(a) + (1 - y) * log10(1 - a))
    
    def train(self, x, y, learning_rate, iterations):
        
        # x: training set of data
        # y: the actual output of the training data
        
        for i in range(iterations):
            
            z_ij = dot(x, self.w_ij) # the dot product of the weights of the hidden layer and the inputs
            a_ij = self.Sigmoid(z_ij) # applying the Sigmoid activation function
            
            z_jk = dot(a_ij, self.w_jk) # the same previous process will be applied to find the predicted output
            a_jk = self.Sigmoid(z_jk)  
            
            dl_jk = -y/a_jk + (1 - y)/(1 - a_jk) # the derivative of the cross entropy loss wrt output
            da_jk = self.SigmoidDerivative(a_jk) # the derivative of Sigmoid  wrt the input (before activ.) of the output layer
            dz_jk = a_ij # the derivative of the inputs of the hidden layer (before activation) wrt weights of the output layer
            
            dl_ij = dot(da_jk * dl_jk, self.w_jk.T) # the derivative of cross entropy loss wrt hidden layer input (after activ.)
            da_ij = self.SigmoidDerivative(a_ij) # the derivative of Sigmoid wrt the inputs of the hidden layer (before activ.)
            dz_ij = x # the derivative of the inputs of the hidden layer (before activation) wrt weights of the hidden layer
            
            # calculating the gradient using the chain rule
            gradient_ij = dot(dz_ij.T , dl_ij * da_ij)
            gradient_jk = dot(dz_jk.T , dl_jk * da_jk)
            
            # calculating the new optimal weights
            self.w_ij = self.w_ij - learning_rate * gradient_ij 
            self.w_jk = self.w_jk - learning_rate * gradient_jk
            
            # printing the loss of our neural network after each 1000 iteration
            if i % 1000 == 0 in range(iterations):
                print("loss: ", self.crossentropyerror(a_jk, y))
                  
    def predict(self, inputs):
        
        # predicting the class of the input data after weights optimization
        
        output_from_layer1 = self.Sigmoid(dot(inputs, self.w_ij)) # the output of the hidden layer
        
        output_from_layer2 = self.Sigmoid(dot(output_from_layer1, self.w_jk)) # the output of the output layer
        
        return output_from_layer1, output_from_layer2
    
    # the function will print the initial starting weights before training
    def SynapticWeights(self):
        
        print("Layer 1 (4 neurons, each with 3 inputs): ")
        
        print("w_ij: ", self.w_ij)
        
        print("Layer 2 (1 neuron, with 4 inputs): ")
        
        print("w_jk: ", self.w_jk)

    
def main():
    
    ANN = ArtificialNeuralNetwork()
    
    ANN.SynapticWeights()
    
    # the training inputs 
    # the last column is used to add non linearity to the clasification task
    x = array([[0, 0, 1], 
               [0, 1, 1], 
               [1, 0, 1], 
               [0, 1, 0], 
               [1, 0, 0], 
               [1, 1, 1], 
               [0, 0, 0]])
    
    # the training outputs
    y = array([[0, 1, 1, 1, 1, 0, 0]]).T

    ANN.train(x, y, 1, 10000)
    
    # Printing the new synaptic weights after training
    print("New synaptic weights after training: ")
    print("w_ij: ", ANN.w_ij)
    print("w_jk: ", ANN.w_jk)
    
    # Our prediction after feeding the ANN with new set of data
    print("Considering new situation [1, 1, 0] -> ?: ")
    print(ANN.predict(array([[1, 1, 0]])))
    
if __name__=="__main__":
    main()

