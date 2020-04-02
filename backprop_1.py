import numpy as np
import mnist
import matplotlib
import matplotlib.pyplot as plt

#questa rete fa una cosa diversa: stabilisce se l'input è uno zero o meno
#quindi posso usare una funzione d'attivazione binaria e cross entropy a
#2 variabili
#la rete sarà con 1 strato hidden con numero variabile di
#neuroni: 784 -> n_hidd -> 1

X_train, Y_train = mnist.train_images(), mnist.train_labels()
X_test, Y_test = mnist.test_images(), mnist.test_labels()

print(X_train.shape)

X_train = X_train.reshape(60000, 28*28)/255
X_test = X_test.reshape(10000, 28*28)/255

a = np.equal(Y_train, 0)
y = np.zeros_like(Y_train)
y[a] = 1
Y_train = y

a = np.equal(Y_test, 0)
y = np.zeros_like(Y_test)
y[a] = 1
Y_test = y

shuffle_index = np.random.permutation(60000)
X_train, Y_train = X_train[shuffle_index, :], Y_train[shuffle_index]

def sigmoid(z):
    s = 1 / (1+ np.exp(-z))
    return s

def relu(z):
    z[z < 0] = 0
    return z

#costruisco la rete
class network():
    def __init__(self, X_train, Y_train, X_test, Y_test, hidd_size = 50, learning_rate = 1e-4):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.input_size = self.X_train.shape[1]
        self.m_train = self.X_train.shape[0]
        self.m_test = self.X_test.shape[0]
        self.hidden_size = 150
        self.output_size = 1
        self.learning_rate = learning_rate

        self.w1 = np.random.rand(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros(self.hidden_size)
        self.w2 = np.random.rand(self.hidden_size) * 0.01
        self.b2 = np.zeros(1)

    def feedforward(self, index, option = "train"):
        if option == "train":
            self.z1 = np.matmul(self.X_train[index], self.w1) + self.b1
        else:
            self.z1 = np.matmul(self.X_test[index], self.w1) + self.b1
        self.a1 = sigmoid(self.z1)

        self.z2 = np.matmul(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(self.z2)

    def backpropagate(self, index):
        dEdz2 = self.a2 - self.Y_train[index]
        dEdw2 = dEdz2 * self.a1
        dEdb2 = dEdz2
        dEda1 = dEdz2 * self.w2

        dEdz1 = dEda1 * self.a1 * (1-self.a1)
        dEdw1 = np.outer(self.X_train[index], dEdz1)
        dEdb1 = dEdz1

        #update
        self.w2 = self.w2 - self.learning_rate * dEdw2
        self.b2 = self.b2 - self.learning_rate * dEdb2
        self.w1 = self.w1 - self.learning_rate * dEdw1
        self.b1 = self.b1 - self.learning_rate * dEdb1

    def calc_error(self, index):
        E = .5 * (self.Y_train[index] - self.a2)**2
        return E

model = network(X_train, Y_train, X_test, Y_test, learning_rate = 1e-4)

order = np.random.permutation(model.m_train)
train_errors = []
validation_errors = []
train = True
counter = 0
for index in order:
    print("iter: " + str(len(train_errors)))
    if train==True:
        model.feedforward(index)
        train_errors.append(model.calc_error(index))
        model.backpropagate(index)
        train = False
    else:
        model.feedforward(index)
        validation_errors.append(model.calc_error(index))
        train = True
    counter +=1
    if (len(validation_errors)>12 and len(train_errors) > 0 and
    #abs(validation_errors[-1] - train_errors[-1]) / abs(validation_errors[-1] + train_errors[-1]) < 1e-5)
    counter>10000):
        break

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
#test
counter = 0
for index in range(model.m_test):
    model.feedforward(index, option = "test")
    print(str(model.calc_error(index)) + " / " + str(model.Y_test[index]) )
print("corrette su 10000 : " + str(counter))
