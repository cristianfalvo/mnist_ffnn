import numpy as np
import matplotlib.pyplot as plt
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
    z[z < 0] = 0
    return z

def relu1(z):
    z[z <= 0] = 0
    z[z > 0] = 1
    return z

def softmax(z):#(30,10)
    b = np.max(z)
    z = np.exp(z - b)
    return z / np.sum(z)

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
        self.w1 = np.random.randn(self.input_size, self.hidd1_size) / np.sqrt(2/(self.input_size + self.hidd1_size)) #(784, 150)
        print(self.w1.shape)
        self.b1 = np.zeros(self.hidd1_size)
        self.w2 = np.random.randn(self.hidd1_size, self.hidd2_size) / np.sqrt(2/(self.hidd1_size + self.hidd2_size))#(150,120)
        self.b2 = np.zeros(self.hidd2_size)
        self.w3 = np.random.randn(self.hidd2_size, self.output_size) / np.sqrt(2/(self.hidd2_size + self.output_size))#(120,10)
        self.b3 = np.zeros(self.output_size)

        #minibatches are only for training and validating
        self.minibatch_size = batch_size
        self.minibatch_number = int(self.n_train_ex / self.minibatch_size)

        self.minibatches_X = np.split(self.X_train, self.minibatch_number)
        self.minibatches_Y = np.split(self.Y_train, self.minibatch_number)

    def feedforward(self, batch_index):
        self.z1 = self.minibatches_X[batch_index] @ self.w1 + self.b1 # (30, 784) @ (784, 150) + (150) -> (30,150)
        self.a1 = relu(self.z1)

        self.z2 = self.a1 @ self.w2 + self.b2 # (30,150) @ (150,120) + (120) -> (30,120)
        self.a2 = relu(self.z2)

        self.z3 = self.a2 @ self.w3 + self.b3 # (30,120) @ (120,10) + (10) -> (30,10)
        self.a3 = softmax(self.z3)
        return self.a3

    def backpropagate(self, batch_index):
        dEdz3 = self.minibatches_Y[batch_index] * (self.a3 - 1) #(30,10)
        dEda2 = np.einsum("ij, jk -> ik", dEdz3, self.w3.T) #(30,10) ein (10,120) -> (30,120)
        #dEdw3 = np.einsum("ij, ik -> ijk", dEdz3, self.a2) #(30,10) ein (30,120) -> (30,10,120)
        dEdw3 = np.einsum("ij, ik -> ikj", dEdz3, self.a2) #(30,10) ein (30,120) -> (30,120,10)
        dEdb3 = dEdz3

        dEdz2 = dEda2 * relu1(self.z2) #(30,10)
        dEda1 = np.einsum("ij, jk -> ik", dEdz2, self.w2.T) #(30,120) ein (120,150) -> (30,150)
        #dEdw2 = np.einsum("ij, ik -> ijk", dEdz2, self.a1)
        dEdw2 = np.einsum("ij, ik -> ikj", dEdz2, self.a1) #(30,120) ein (30,150) -> (30,150, 120)
        dEdb2 = dEdz2

        dEdz1 = dEda1 * relu1(self.z1) #(30,150)
        #
        #dEdw1 = np.einsum("ij, ik -> ijk", dEdz1, self.minibatches_X[batch_index])
        dEdw1 = np.einsum("ij, ik -> ikj", dEdz1, self.minibatches_X[batch_index]) #(30,150) ein (30,784) -> (30,784,150)
        dEdb1 = dEdz1

        # update params
        self.w3 -= self.learning_rate * np.sum(dEdw3, axis = 0)
        self.w2 -= self.learning_rate * np.sum(dEdw2, axis = 0)
        self.w1 -= self.learning_rate * np.sum(dEdw1, axis = 0)

        self.b3 -= self.learning_rate * np.sum(dEdb3, axis = 0)
        self.b2 -= self.learning_rate * np.sum(dEdb2, axis = 0)
        self.b1 -= self.learning_rate * np.sum(dEdb1, axis = 0)

    def calc_error(self, batch_index):
        E = np.sum(-1 * self.minibatches_Y[batch_index] * np.log(self.a3), axis = 1)
        return np.average(E)

#model= network(X_train, Y_train, X_test, Y_test, [150,150])
#model.feedforward(1)
#model.backpropagate(1)

z = np.array([1, 0, -1, 2, 5, -3]).reshape((6,1))
print(relu(z))
print(relu1(z))

model = network(X_train, Y_train, X_test, Y_test, [150,120])
order = np.random.permutation(model.minibatch_number)
train = True
train_errors = []
validation_errors = []
#pessima implementazione di un'epoch ma non avevo voglia di fare cose più complicate
for batch in order:
    print("Iteration: " + str(batch))
    if train:
        model.feedforward(batch)
        err  = model.calc_error(batch)
        train_errors.append(err)
        model.backpropagate(batch)
        train = False
    else:
        model.feedforward(batch)
        err = model.calc_error(batch)
        validation_errors.append(err)
        train = True

t = np.asarray(train_errors)
v = np.asarray(validation_errors)
normalized_errors = np.abs(t-v) / np.abs(t+v)

fig, axs = plt.subplots(2)
axs[0].plot(train_errors, "b")
axs[0].plot(validation_errors, "r")
axs[1].plot(normalized_errors)
plt.show()
