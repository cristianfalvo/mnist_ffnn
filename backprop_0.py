import numpy as np
import matplotlib as matplotlib
import mnist

X_train = mnist.train_images()
Y_train = mnist.train_labels()

X_test = mnist.test_images()
Y_test = mnist.test_labels()

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]**2))/255
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]**2))/255
print(X_train.shape)

y = np.eye(10)[Y_train.astype('int32')]
Y_train = y
y = np.eye(10)[Y_test.astype('int32')]
Y_test = y

def relu(z):
    return max(0,z)

def relu1(z):
    return np.where(z>0, z, 0)

def softmax(z):#(30,10)
    b= np.amax(z, axis = 1)
    s = np.exp(z-b.T)
    print(z.shape)
    print(b.shape)
    print(s.shape)


class network:
    def __init__(self, X_train, Y_train, X_test, Y_test, hidden_layers, learning_rate = 1, batch_size = 30):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.input_size = self.X_train.shape[1]
        self.output_size = self.Y_train.shape[1]
        self.hidd1_size = hidden_layers[0]
        self.hidd2_size = hidden_layers[1]
        self.n_train_ex = self.X_train.shape[0]
        self.n_test_ex = self.X_test.shape[0]
        self.learning_rate = learning_rate

        #weights and biases
        self.w1 = np.random.randn(self.hidd1_size, self.input_size) / np.sqrt(2/(self.input_size + self.hidd1_size)) #(150, 784)
        print(self.w1.shape)
        self.b1 = np.zeros(self.hidd1_size)
        self.w2 = np.random.randn(self.hidd2_size, self.hidd1_size) / np.sqrt(2/(self.hidd1_size + self.hidd2_size))#(150,150)
        self.b1 = np.zeros(self.hidd2_size)
        self.w1 = np.random.randn(self.output_size, self.hidd2_size) / np.sqrt(2/(self.hidd2_size + self.output_size))#(10,150)
        self.b1 = np.zeros(self.output_size)

        #minibatches are only for training and validating
        self.minibatch_size = batch_size
        self.minibatch_number = int(self.n_train_ex / self.minibatch_size)

        self.minibatches_X = np.split(self.X_train, self.minibatch_number)
        self.minibatches_Y = np.split(self.Y_train, self.minibatch_number)




    def feedforward(self, batch_index):

        self.z1 = self.minibatches_X[batch_index] @ self.w1.T + self.b1 # (30, 784) @ (784, 150) + (150) -> (30,150)
        self.a1 = relu(z1)

        self.z2 = self.a1 @ self.w2.T + self.b2
        self.a2 = relu(z2)

        self.z3 = self.a2 @ self.w3.T + self.b3
        self.a3 = softmax(self.z3)

    def backpropagate(self, batch_index):

        dEdz3 = self.minibatches_Y[batch_index] * (self.a3 - 1) #(30,10)
        dEda2 = np.einsum("ij, jk -> ik", dEdz3, self.w3) #(30,10) ein (10,150) -> (30,150)
        dEdw3 = np.einsum("ij, ik -> ijk", dEdz3, self.a2) #(30,10) ein (30,150) -> (30,10,150)
        dEdb3 = dEdz3


        dEdz2 = dEda2 * relu1(self.z2) #(30,10)
        dEda1 = np.einsum("ij, jk -> ik", dEdz2, self.w2) #(30,10) ein (10,150) -> (30,150)
        dEdw2 = np.einsum("ij, ik -> ijk", dEdz2, self.a2) #(30,10) ein (30,150) -> (30,10,150)
        dEdb2 = dEdz2

        dEdz1 = dEda1 * relu1(self.z1)
        #
        dEdw1 = np.einsum("ij, ik -> ijk", dEdz1, self.a1) #(30,10) ein (30,150) -> (30,10,150)
        dEdb1 = dEdz1

        #update params

        self.w3 -= self.learning_rate * dEdw3
        self.w2 -= self.learning_rate * dEdw2
        self.w1 -= self.learning_rate * dEdw1

        self.b3 -= self.learning_rate * dEdb3
        self.b2 -= self.learning_rate * dEdb2
        self.b1 -= self.learning_rate * dEdb1

model= network(X_train, Y_train, X_test, Y_test, [150,150])
model.feedforward(1)
model.backpropagate(1)

'Babudinibich'