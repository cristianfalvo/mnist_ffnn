import numpy as np
import sys
import matplotlib.pyplot as plt
import mnist
np.set_printoptions(threshold=sys.maxsize)

X_train = mnist.train_images()
Y_train = mnist.train_labels()

X_test = mnist.test_images()
Y_test = mnist.test_labels()

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]**2))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]**2))

y = np.eye(10)[Y_train.astype('int32')]
Y_train = y
y = np.eye(10)[Y_test.astype('int32')]
Y_test = y

def relu(z):
    z[z < 0] = 0
    return z

def sigmoid(z):
    s = 1 / (1+ np.exp(-z))
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

'''def softmax_batch(x):
    """ x has shape (batch size, channel size) """
    x = np.exp(x)
    return x / x.sum(axis=1)[:, np.newaxis]'''


def softmax(z): #(10)
    b=np.max(z)
    s = np.exp(z-b)
    return s / np.sum(s)

def delta(i, j):
    if i==j:
        return 1
    else:
        return 0

class network:
    def __init__(self, X_train, Y_train, X_test, Y_test, hidden_layers, learning_rate = 1e-4, batch_size = 30):
        self.X_train = X_train/255
        self.Y_train = Y_train
        self.X_test = X_test/255
        self.Y_test = Y_test
        self.input_size = self.X_train.shape[1]
        self.output_size = self.Y_train.shape[1]
        self.hidd1_size = hidden_layers[0]
        self.hidd2_size = hidden_layers[1]
        self.n_train_ex = self.X_train.shape[0]
        self.n_test_ex = self.X_test.shape[0]
        self.learning_rate = learning_rate
        self.check = False

        #weights and biases
        self.w1 = np.random.rand(self.input_size, self.hidd1_size) * np.sqrt(2/(self.input_size + self.hidd1_size)) #(784, 150)
        #print(self.w1)
        #print(self.w1.shape)
        self.b1 = np.zeros(self.hidd1_size)
        self.w2 = np.random.rand(self.hidd1_size, self.hidd2_size) * np.sqrt(2/(self.hidd1_size + self.hidd2_size))#(150,120)
        self.b2 = np.zeros(self.hidd2_size)
        self.w3 = np.random.rand(self.hidd2_size, self.output_size) * np.sqrt(2/(self.hidd2_size + self.output_size))#(120,10)
        self.b3 = np.zeros(self.output_size)

        #minibatches are only for training and validating
        self.minibatch_size = batch_size
        self.minibatch_number = int(self.n_train_ex / self.minibatch_size)

        self.minibatches_X = np.split(self.X_train, self.minibatch_number)
        self.minibatches_Y = np.split(self.Y_train, self.minibatch_number)

    def feedforward(self, index, option = "train"):
        #self.z1 = self.minibatches_X[batch_index] @ self.w1 + self.b1 # (30, 784) @ (784, 150) + (150) -> (30,150)
        if option=="train":
            self.z1 = np.dot(self.X_train[index], self.w1) + self.b1 # (30, 784) @ (784, 150) + (150) -> (30,150)
        else:
            self.z1 = np.dot(self.X_test[index], self.w1) + self.b1
        #self.a1 = relu(self.z1)
        self.a1 = sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.w2) + self.b2 # (30,150) @ (150,120) + (120) -> (30,120)
        #self.a2 = relu(self.z2)
        self.a2 = sigmoid(self.z2)

        self.z3 = np.dot(self.a2 , self.w3) + self.b3 # (30,120) @ (120,10) + (10) -> (30,10)
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
        dEdz3 =self.a3 - self.minibatches_Y[batch_index] #(30,10)
        dEda2 = np.matmul( dEdz3, self.w3.T) #(30,10) ein (10,120) -> (30,120)
        #dEda2 = np.einsum("ij, jk -> ik", dEdz3, self.w3.T) #(30,10) ein (10,120) -> (30,120)
        #dEdw3 = np.einsum("ij, ik -> ijk", dEdz3, self.a2) #(30,10) ein (30,120) -> (30,10,120)
        dEdw3 = np.einsum("ij, ik -> ikj", dEdz3, self.a2) #(30,10) ein (30,120) -> (30,120,10)
        dEdb3 = dEdz3

        #dEdz2 = dEda2 * self.a2 * (1 - self.a2)#(30,10)
        dEdz2 = dEda2 * relu1(self.z2)#(30,10)
        dEda1 = np.matmul(dEdz2, self.w2.T) #(30,120) ein (120,150) -> (30,150)
        #dEdw2 = np.einsum("ij, ik -> ijk", dEdz2, self.a1)
        dEdw2 = np.einsum("ij, ik -> ikj", dEdz2, self.a1) #(30,120) ein (30,150) -> (30,150, 120)
        dEdb2 = dEdz2

        #dEdz1 = dEda1 * self.a1 * (1 - self.a1) #(30,150)
        dEdz1 = dEda1 * relu1(self.z1) #(30,150)

        #dEdw1 = np.einsum("ij, ik -> ijk", dEdz1, self.minibatches_X[batch_index])
        dEdw1 = np.einsum("ij, ik -> ikj", dEdz1, self.minibatches_X[batch_index]) #(30,150) ein (30,784) -> (30,784,150)
        dEdb1 = dEdz1
        # update params
        self.w3 -= 1/self.minibatch_size * self.learning_rate * np.sum(dEdw3, axis = 0)
        self.w2 -= 1/self.minibatch_size * self.learning_rate * np.sum(dEdw2, axis = 0)
        self.w1 -= 1/self.minibatch_size * self.learning_rate * np.sum(dEdw1, axis = 0)

        self.b3 -= 1/self.minibatch_size * self.learning_rate * np.sum(dEdb3, axis = 0)
        self.b2 -= 1/self.minibatch_size * self.learning_rate * np.sum(dEdb2, axis = 0)
        self.b1 -= 1/self.minibatch_size * self.learning_rate * np.sum(dEdb1, axis = 0)

        '''print("update/weight ratio:")
        print(np.average(1/self.minibatch_size * self.learning_rate * np.sum(dEdw3, axis = 0)/ self.w3))
        print(np.average(1/self.minibatch_size * self.learning_rate * np.sum(dEdw2, axis = 0)/ self.w2))
        print(np.average(1/self.minibatch_size * self.learning_rate * np.sum(dEdw1, axis = 0)/ self.w1))'''

        '''print("weight average")
        print(np.average(self.w3))
        print(np.average(self.w2))
        print(np.average(self.w1))
'''
        '''        print("check")
        print(np.sum(self.a3, axis = 1))
        '''
# for numerical gradient check only
    def cross_entropy_prime(self, y, a, eps = 1e-6):
        out = np.zeros_like(a)
        for i in range(out.shape[0]):
            upper = 0
            for j in range(y.shape[0]):
                upper -= y[j] * np.log(a[j] + eps * delta(i, j))
            lower = 0
            for j in range(y.shape[0]):
                lower -= y[j] * np.log(a[j] - eps * delta(i, j))
            out[i] = (upper - lower) / (2*eps)
        return out
# for numerical gradient check only
    def softmax_prime(self, z, eps = 1e-6):
        out = np.zeros_like(z)
        for i in range(out.shape[0]):
            delta = np.zeros_like(z)
            delta[i] = eps
            upper = softmax(z + delta)[i]
            lower = softmax(z - delta)[i]
            out[i] = (upper - lower) / (2*eps)
        return out

    def backpropagate(self, index):
        dEda3 = - self.Y_train[index] * 1/self.a3
        #dEdz3 = self.Y_train[index] * (self.a3 - 1) #(10)
        '''dEda3_explicit = self.foo(self.Y_train[index], self.a3)
        if np.any(abs(dEda3 - dEda3_explicit)) > 1e-7:
            print("a3 =\n" + str(self.a3))
            print("Y_train =\n" + str(self.Y_train[index]))
            print("Analytical gradient: " + str(dEda3))
            print("explicit gradient: " + str(dEda3_explicit))
            print("gradient difference: " + str(abs(dEda3 - dEda3_explicit)))
            quit()'''
        da3dz3 = self.a3 * (1-self.a3)
        '''da3dz3_explicit = self.softmax_prime(self.z3)
        if np.any(abs(da3dz3 - da3dz3_explicit)) > 1e-7:
            print("z3 =\n" + str(self.z3))
            print("Analytical gradient: " + str(da3dz3))
            print("explicit gradient: " + str(da3dz3_explicit))
            print("gradient difference: " + str(abs(da3dz3 - da3dz3_explicit)))
            print(dEda3 * da3dz3)
            quit()'''

        dEdz3 = dEda3 * da3dz3 #(10)
        dEda2 = np.matmul(dEdz3,self.w3.T) # (120, 10) * (10) = (120)
        dEdw3 = np.outer(self.a2, dEdz3) # (120) * (10) -> (120, 10)
        dEdb3 = dEdz3

        #dEdz2 = dEda2 * relu1(self.z2)
        dEdz2 = dEda2 * self.a2 * (1-self.a2)
        dEda1 = np.matmul(dEdz2,self.w2.T)
        dEdw2 = np.outer(self.a1, dEdz2)
        dEdb2 = dEdz2

        #dEdz1 = dEda1 * relu1(self.z1)
        dEdz1 = dEda1 * self.a1 * (1-self.a1)
        dEdb1 = dEdz1
        dEdw1 = np.outer(self.X_train[index], dEdz1)

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
        if (np.any(np.equal(self.a3, 0))):
            print("a3")
            print(self.a3)
            print("z3")
            print(self.z3)
            print("a2")
            print(self.a2)
            print("z2")
            print(self.z2)
            print("a1")
            print(self.a1)
            print("z1")
            print(self.z1)
            print("input")
            print(self.minibatches_X[batch_index])
            self.check = True
        E = -1 * np.sum(self.minibatches_Y[batch_index] * np.log(self.a3 + 1e-10), axis = 1)
        return np.average(E)

#model= network(X_train, Y_train, X_test, Y_test, [150,150])
#model.feedforward(1)
#model.backpropagate(1)

model = network(X_train, Y_train, X_test, Y_test, [150,120])
'''model.feedforward(10)
print(model.calc_error(10))
model.backpropagate(10)
input()'''

def batch():
    train = True
    train_errors = []
    validation_errors = []

    for i in range(30):
        order =  np.random.permutation(model.minibatch_number)
        counter = 0
        for batch in order:
            print("Iteration: " + str(counter))
            counter += 1
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
            if model.check:
                input()
        if model.check:
            break
        print("epoch " + str(i) + " finished")

    t = np.asarray(train_errors)
    v = np.asarray(validation_errors)
    if t.shape[0] > v.shape[0]:
        t = t[:t.shape[0]-1]
    normalized_errors = np.abs(t-v) / np.abs(t+v)

    fig, axs = plt.subplots(2)
    axs[0].plot(train_errors, "b")
    axs[0].plot(validation_errors, "r")
    axs[1].plot(normalized_errors)
    plt.show()

    input()
    counter = 0
    for index in range(model.n_test_ex):
        model.feedforward(index, option = "test")
        print(str(np.argmax(model.a3)) + " -> " + str(np.argmax(model.Y_test[index])))
        if (np.argmax(model.a3) == np.argmax(model.Y_test[index])):
            counter +=1
    print("corrette su 10000 : " + str(counter))

def non_batch():
    #non batch learning
    train_errors = []
    validation_errors = []
    order = np.random.permutation(model.n_train_ex)
    print(model.n_train_ex)
    train = True
    for batch in order:
        print("Example: " + str(batch))
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

    input()
    counter = 0
    for index in range(model.n_test_ex):
        model.feedforward(index, option = "test")
        print(str(np.argmax(model.a3)) + " -> " + str(np.argmax(model.Y_test[index])))
        if (np.argmax(model.a3) == np.argmax(model.Y_test[index])):
            counter +=1
    print("corrette su 10000 : " + str(counter))

batch()
