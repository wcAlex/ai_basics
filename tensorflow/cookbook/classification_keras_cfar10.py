# apply all kinds initalization, optimization and regularization.
import tensorflow as tf
from tensorflow import keras

from matplotlib import pyplot
from tensorflow.keras.datasets import cifar10

import numpy as np

import os
import matplotlib as mpl
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)

# load dataset
(trainX, trainy), (testX, testy) = cifar10.load_data()
# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))

X_valid, X_train = trainX[:5000] / 255., trainX[5000:] / 255.
y_valid, y_train = trainy[:5000], trainy[5000:]
X_test, y_test = testX / 255., testy

X_means = X_train.mean(axis=0)
X_stds = X_train.std(axis=0)
X_train_scaled = (X_train - X_means) / X_stds
X_valid_scaled = (X_valid - X_means) / X_stds
X_test_scaled = (X_test - X_means) / X_stds

def plotImages(trainX):
    # plot first few images
    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # plot raw pixel data
        pyplot.imshow(trainX[i])
    # show the figure
    pyplot.savefig("sample.jpg")


#plotImages(trainX)

# define architecture
def build_model(n_hidden=1, n_neurons=30, input_shape=[8]):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    #model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))

    for layer in range(n_hidden):
        #model.add(keras.layers.Dense(n_neurons, kernel_initializer='he_normal'))

        # selu activation function requires lecun_normal initializer.
        model.add(keras.layers.Dense(n_neurons, kernel_initializer='lecun_normal'))

        model.add(keras.layers.BatchNormalization())
        #model.add(keras.layers.Activation('elu'))
        # selu requires normalized input.
        model.add(keras.layers.Activation('selu'))

    model.add(keras.layers.AlphaDropout(rate=0.1))
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    return model

# Batch Norm with elu activation function on keras.optimizers.Nadam(lr=5e-4) is the best in this code.

# model = keras.models.Sequential()
# model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
# model.add(keras.layers.BatchNormalization())
# for _ in range(20):
#     model.add(keras.layers.Dense(100, kernel_initializer="he_normal"))
#     model.add(keras.layers.BatchNormalization())
#     model.add(keras.layers.Activation("elu"))
# model.add(keras.layers.Dense(10, activation="softmax"))


model = build_model(n_hidden=20, n_neurons=100, input_shape=trainX.shape[1:])
# one way to select the best learning rate is to try different learning rate, 
# ex: 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3 and 1e-2
learning_rate_in_run = 5e-4
optimizer = keras.optimizers.Nadam(lr=learning_rate_in_run)

print(model.summary())

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])

# define call backs, tensorboard, early stoping, model saving, 
root_logdir = os.path.join(os.curdir, "my_cfar10_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S") + "_" + str(learning_rate_in_run)
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()

early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
checkpoint_cb = keras.callbacks.ModelCheckpoint("keras_seq_cfar10_model.h5", save_best_only=True)

callbacks = [early_stopping_cb, checkpoint_cb, tensorboard_cb]

# search learning rate.
K = keras.backend

class ExponentialLearningRate(keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []
    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.lr))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)

def find_learning_rate(model, X, y, epochs=1, batch_size=32, min_rate=10**-5, max_rate=10):
    init_weights = model.get_weights()
    iterations = len(X) // batch_size * epochs
    factor = np.exp(np.log(max_rate / min_rate) / iterations)
    init_lr = K.get_value(model.optimizer.lr)
    K.set_value(model.optimizer.lr, min_rate)
    exp_lr = ExponentialLearningRate(factor)
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size,
                        callbacks=[exp_lr])
    K.set_value(model.optimizer.lr, init_lr)
    model.set_weights(init_weights)
    return exp_lr.rates, exp_lr.losses

def plot_lr_vs_loss(rates, losses):
    plt.plot(rates, losses)
    plt.gca().set_xscale('log')
    plt.hlines(min(losses), min(rates), max(rates))
    plt.axis([min(rates), max(rates), min(losses), (losses[0] + min(losses)) / 2])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    plt.savefig('lr_vs_loss.jpg')

def search_best_lr():
    batch_size = 128
    rates, losses = find_learning_rate(model, X_train, y_train, epochs=1, batch_size=batch_size)
    plot_lr_vs_loss(rates, losses)

#search_best_lr()

def train():
    model.fit(X_train_scaled, y_train, epochs=100, validation_data=(X_valid_scaled, y_valid), callbacks=callbacks)
    print(model.evaluate(X_test_scaled, y_test))

train()
# MCDroput: prediction on a trained model

class MCAlphaDropout(keras.layers.AlphaDropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

mc_model = keras.models.Sequential([
    MCAlphaDropout(layer.rate) if isinstance(layer, keras.layers.AlphaDropout) else layer
    for layer in model.layers
]) 

def mc_dropout_predict_probas(mc_model, X, n_samples=10):
    Y_probs = [ mc_model.predict(X) for sample in range(n_samples)]
    return np.mean(Y_probs, axis=0)

def mc_dropout_predict_classes(mc_model, X, n_samples=10):
    Y_probs = mc_dropout_predict_probas(mc_model, X, n_samples)
    return np.argmax(Y_probs, axis=1)

y_pred = mc_dropout_predict_classes(mc_model, X_valid_scaled)
accuracy = np.mean(y_pred == y_valid[:, 0])
print(accuracy)

# n_epoch = 1
# history = model.fit(X_train, y_train, epochs=n_epoch, validation_data=(X_valid, y_valid),
#     callbacks=[early_stopping_cb, tensorboard_cb, checkpoint_cb])

class OneCycleScheduler(keras.callbacks.Callback):
    def __init__(self, iterations, max_rate, start_rate=None,
                 last_iterations=None, last_rate=None):
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_rate = last_rate or self.start_rate / 1000
        self.iteration = 0
    def _interpolate(self, iter1, iter2, rate1, rate2):
        return ((rate2 - rate1) * (self.iteration - iter1)
                / (iter2 - iter1) + rate1)
    def on_batch_begin(self, batch, logs):
        if self.iteration < self.half_iteration:
            rate = self._interpolate(0, self.half_iteration, self.start_rate, self.max_rate)
        elif self.iteration < 2 * self.half_iteration:
            rate = self._interpolate(self.half_iteration, 2 * self.half_iteration,
                                     self.max_rate, self.start_rate)
        else:
            rate = self._interpolate(2 * self.half_iteration, self.iterations,
                                     self.start_rate, self.last_rate)
            rate = max(rate, self.last_rate)
        self.iteration += 1
        K.set_value(self.model.optimizer.lr, rate)


def train_onecycle():
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
    for _ in range(20):
        model.add(keras.layers.Dense(100,
                                    kernel_initializer="lecun_normal",
                                    activation="selu"))

    model.add(keras.layers.AlphaDropout(rate=0.1))
    model.add(keras.layers.Dense(10, activation="softmax"))

    optimizer = keras.optimizers.SGD(lr=1e-2)
    model.compile(loss="sparse_categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"])

    n_epochs = 15
    onecycle = OneCycleScheduler(len(X_train_scaled) // batch_size * n_epochs, max_rate=0.05)
    history = model.fit(X_train_scaled, y_train, epochs=n_epochs, batch_size=batch_size,
                        validation_data=(X_valid_scaled, y_valid),
                        callbacks=[onecycle])

    


