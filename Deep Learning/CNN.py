
# coding: utf-8

# In[1]:


# Import the deep learning library
import tensorflow as tf
import time

# Import the MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Network inputs and outputs
# The network's input is a 28×28 dimensional input
n = 28
m = 28
num_input = n * m # MNIST data input 
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])

# Storing the parameters of our LeNET-5 inspired Convolutional Neural Network
weights = {
   "W_ij": tf.Variable(tf.random_normal([5, 5, 1, 32])),
   "W_jk": tf.Variable(tf.random_normal([5, 5, 32, 64])),
   "W_kl": tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
   "W_lm": tf.Variable(tf.random_normal([1024, num_classes]))
    }

biases = {
   "b_ij": tf.Variable(tf.random_normal([32])),
   "b_jk": tf.Variable(tf.random_normal([64])),
   "b_kl": tf.Variable(tf.random_normal([1024])),
   "b_lm": tf.Variable(tf.random_normal([num_classes]))
    }

# The hyper-parameters of our Convolutional Neural Network
learning_rate = 1e-3
num_steps = 500
batch_size = 128
display_step = 10

def ConvolutionLayer(x, W, b, strides=1):
    # Convolution Layer
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return x

def ReLU(x):
    # ReLU activation function
    return tf.nn.relu(x)

def PoolingLayer(x, k=2, strides=2):
    # Max Pooling layer
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, strides, strides, 1],
                          padding='SAME')

def Softmax(x):
    # Softmax activation function for the CNN's final output
    return tf.nn.softmax(x)

# Create model
def ConvolutionalNeuralNetwork(x, weights, biases):
    # MNIST data input is a 1-D row vector of 784 features (28×28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    Conv1 = ConvolutionLayer(x, weights["W_ij"], biases["b_ij"])
    # Non-Linearity
    ReLU1 = ReLU(Conv1)
    # Max Pooling (down-sampling)
    Pool1 = PoolingLayer(ReLU1, k=2)

    # Convolution Layer
    Conv2 = ConvolutionLayer(Pool1, weights["W_jk"], biases["b_jk"])
    # Non-Linearity
    ReLU2 = ReLU(Conv2)
    # Max Pooling (down-sampling)
    Pool2 = PoolingLayer(ReLU2, k=2)
    
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    FC = tf.reshape(Pool2, [-1, weights["W_kl"].get_shape().as_list()[0]])
    FC = tf.add(tf.matmul(FC, weights["W_kl"]), biases["b_kl"])
    FC = ReLU(FC)

    # Output, class prediction
    output = tf.add(tf.matmul(FC, weights["W_lm"]), biases["b_lm"])
    
    return output

# Construct model
logits = ConvolutionalNeuralNetwork(X, weights, biases)
prediction = Softmax(logits)

# Softamx cross entropy loss function
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))

# Optimization using the Adam Gradient Descent optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_process = optimizer.minimize(loss_function)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# recording how the loss functio varies over time during training
cost = tf.summary.scalar("cost", loss_function)
training_accuracy = tf.summary.scalar("accuracy", accuracy)
train_summary_op = tf.summary.merge([cost,training_accuracy])

train_writer = tf.summary.FileWriter("./Desktop/logs",
                                        graph=tf.get_default_graph())

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    
    start_time = time.time()
    
    for step in range(1, num_steps+1):
        
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(training_process, feed_dict={X: batch_x, Y: batch_y})
        
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc, summary = sess.run([loss_function, accuracy, train_summary_op], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            train_writer.add_summary(summary, step)
            
            print("Step " + str(step) + ", Minibatch Loss= " +                   "{:.4f}".format(loss) + ", Training Accuracy= " +                   "{:.3f}".format(acc))
            
    end_time = time.time() 
    
    print("Time duration: " + str(int(end_time-start_time)) + " seconds")
    print("Optimization Finished!")
            
    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:",         sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                      Y: mnist.test.labels[:256]}))

