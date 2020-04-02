import numpy as np
import matplotlib.pyplot as plt
import mnist

X_train = mnist.train_images()
Y_train = mnist.train_labels()

X_test = mnist.test_images()
Y_test = mnist.test_labels()

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]**2))/255
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]**2))/255

y = np.eye(10)[Y_train.astype('int32')]
Y_train = y
y = np.eye(10)[Y_test.astype('int32')]
Y_test = y

def relu(z):
    z[z < 0] = 0
    return z

def sigmoid(z):
    s = 1 / (1+ np.exp(-1*z))
    return s

def relu1(z):
    z[z <= 0] = 0
    z[z > 0] = 1
    return z

def softmax_batch(z):#(30,10)
    b = np.max(z, axis = 1) #(30)
    z = np.exp(z.T - b).T #(30,10)
    out = z.T / np.sum(z, axis = 1) #(10,30)/(30)
    return out.T


def softmax(z): #(10)
    b=np.max(z)
    s = np.exp(z-b)
    return s / np.sum(s)

class network:
    def __init__(self, X_train, Y_train, X_test, Y_test, hidden_layers, learning_rate = 1e-4, batch_size = 30):
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
        self.w1 = np.random.randn(self.input_size, self.hidd1_size) * np.sqrt(2/(self.input_size + self.hidd1_size)) #(784, 150)
        print(self.w1)
        #print(self.w1.shape)
        self.b1 = np.zeros(self.hidd1_size)
        self.w2 = np.random.randn(self.hidd1_size, self.hidd2_size) * np.sqrt(2/(self.hidd1_size + self.hidd2_size))#(150,120)
        self.b2 = np.zeros(self.hidd2_size)
        self.w3 = np.random.randn(self.hidd2_size, self.output_size) * np.sqrt(2/(self.hidd2_size + self.output_size))#(120,10)
        self.b3 = np.zeros(self.output_size)

        #minibatches are only for training and validating
        self.minibatch_size = batch_size
        self.minibatch_number = int(self.n_train_ex / self.minibatch_size)

        self.minibatches_X = np.split(self.X_train, self.minibatch_number)
        self.minibatches_Y = np.split(self.Y_train, self.minibatch_number)

    def feedforward(self, batch_index, option = "train"):
        #self.z1 = self.minibatches_X[batch_index] @ self.w1 + self.b1 # (30, 784) @ (784, 150) + (150) -> (30,150)
        if option=="train":
            self.z1 = self.X_train[batch_index] @ self.w1 + self.b1 # (30, 784) @ (784, 150) + (150) -> (30,150)
        else:
            self.z1 = self.X_test[batch_index] @ self.w1 + self.b1
        #self.a1 = relu(self.z1)
        self.a1 = relu(self.z1)

        self.z2 = self.a1 @ self.w2 + self.b2 # (30,150) @ (150,120) + (120) -> (30,120)
        #self.a2 = relu(self.z2)
        self.a2 = relu(self.z2)

        self.z3 = self.a2 @ self.w3 + self.b3 # (30,120) @ (120,10) + (10) -> (30,10)
        self.a3 = softmax(self.z3)
        #print(self.a3)
        return self.a3

    def feedforward_batch(self, batch_index):
        self.z1 = self.minibatches_X[batch_index] @ self.w1 + self.b1 # (30, 784) @ (784, 150) + (150) -> (30,150)
        #self.a1 = relu(self.z1)
        self.a1 = relu(self.z1)

        self.z2 = self.a1 @ self.w2 + self.b2 # (30,150) @ (150,120) + (120) -> (30,120)
        #self.a2 = relu(self.z2)
        self.a2 = relu(self.z2)

        self.z3 = self.a2 @ self.w3 + self.b3 # (30,120) @ (120,10) + (10) -> (30,10)
        self.a3 = softmax_batch(self.z3)
        #print(self.a3)
        return self.a3

    def backpropagate_batch(self, batch_index):
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

        print("update/weight ratio:")
        print(np.average(1/self.minibatch_size * self.learning_rate * np.sum(dEdw3, axis = 0)/ self.w3))
        print(np.average(1/self.minibatch_size * self.learning_rate * np.sum(dEdw2, axis = 0)/ self.w2))
        print(np.average(1/self.minibatch_size * self.learning_rate * np.sum(dEdw1, axis = 0)/ self.w1))

        print("weight average")
        print(np.average(self.w3))
        print(np.average(self.w2))
        print(np.average(self.w1))

        '''        print("check")
        print(np.sum(self.a3, axis = 1))
        '''
    def backpropagate(self, index, option = "train"):
        if option == "train":
            dEdz3 = self.Y_train[index] * (self.a3 - 1) #(10)
        else:
            dEdz3 = self.Y_test[index] * (self.a3 - 1) # (10)
        dEda2 = np.matmul(self.w3, dEdz3) # (120, 10) * (10) = (120)
        dEdw3 = np.outer(self.a2, dEdz3) # (120) * (10) -> (120, 10)
        dEdb3 = dEdz3

        #dEdz2 = dEda2 * relu1(self.z2)
        dEdz2 = dEda2 * self.a2 * (1-self.a2)
        dEda1 = np.matmul(self.w2, dEdz2)
        dEdw2 = np.outer(self.a1, dEdz2)
        dEdb2 = dEdz2

        #dEdz1 = dEda1 * relu1(self.z1)
        dEdz1 = dEda1 * self.a1 * (1-self.a1)
        dEdb1 = dEdz1
        if option=="train":
            dEdw1 = np.outer(self.X_train[index], dEdz1)
        else:
            dEdw1 = np.outer(self.X_test[index], dEdz1)

        self.w3 -= self.learning_rate * dEdw3
        self.w2 -= self.learning_rate * dEdw2
        self.w1 -= self.learning_rate * dEdw1

        self.b3 -= self.learning_rate * dEdb3
        self.b2 -= self.learning_rate * dEdb2
        self.b1 -= self.learning_rate * dEdb1

    def calc_error(self, index):
        E = -1 * np.sum(self.Y_train[index] * np.log(self.a3))
        return E

    def calc_error_batch(self, batch_index):
        E = -1 * np.sum(self.minibatches_Y[batch_index] * np.log(self.a3), axis = 1)
        return np.average(E)

#model= network(X_train, Y_train, X_test, Y_test, [150,150])
#model.feedforward(1)
#model.backpropagate(1)

model = network(X_train, Y_train, X_test, Y_test, [150,120])

train = True
train_errors = []
validation_errors = []

for i in range(3):
    order =  np.random.permutation(model.minibatch_number)
    for batch in order:
        print("Batch: " + str(batch))
        if train:
            model.feedforward_batch(batch)
            err  = model.calc_error_batch(batch)
            train_errors.append(err)
            model.backpropagate_batch(batch)
            train = False
        else:
            model.feedforward_batch(batch)
            err = model.calc_error_batch(batch)
            validation_errors.append(err)
            train = True
    print("epoch " + str(i) + " finished")

t = np.asarray(train_errors)
v = np.asarray(validation_errors)
normalized_errors = np.abs(t-v) / np.abs(t+v)

fig, axs = plt.subplots(2)
axs[0].plot(train_errors, "b")
axs[0].plot(validation_errors, "r")
axs[1].plot(normalized_errors)
plt.show()

quit()
#non batch learning
order = np.random.permutation(model.n_train_ex)
print(model.n_train_ex)

for batch in order:
    print("Batch: " + str(batch))
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
