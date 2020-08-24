import tensorflow as tf
from tensorflow import keras

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

import matplotlib as mpl
import matplotlib.pyplot as plt

import os

# prepare dataset
#fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()

X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.

# plt.imshow(X_train[0], cmap="binary")
# plt.axis('off')
# plt.show()

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


print (X_train.shape, y_train.shape)
print (class_names[y_train[0]])

# define model architecture
def build_model():
    # model = keras.models.Sequential([
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(300, activation="relu"),
    #     keras.layers.Dense(100, activation="relu"),
    #     keras.layers.Dense(10, activation="softmax")
    # ])

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])

    return model

# define callbacks
early_stopping_cb = keras.callbacks.EarlyStopping(patience=20,
                                                  restore_best_weights=True)

root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()

class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))

tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
checkpoint_cb = keras.callbacks.ModelCheckpoint("keras_seq_minst_model.h5", save_best_only=True)
val_train_ratio_cb = PrintValTrainRatioCallback()

# training
model = build_model()
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=2e-1),
              metrics=["accuracy"])

history = model.fit(X_train, y_train, batch_size=30, epochs=150, 
    validation_data=(X_valid, y_valid), callbacks=[val_train_ratio_cb, checkpoint_cb, tensorboard_cb, early_stopping_cb])

# #analyize result
# print(history.epoch, history.params)

model.evaluate(X_test, y_test)

# Optimizing
# find properiate learning_rate
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

def explore_properiate_lr():
    model = build_model()
    model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=1e-3),
              metrics=["accuracy"])

    expon_lr = ExponentialLearningRate(factor=1.005)
    history = model.fit(X_train, y_train, epochs=1,
                    validation_data=(X_valid, y_valid),
                    callbacks=[expon_lr])

    plt.plot(expon_lr.rates, expon_lr.losses)
    plt.gca().set_xscale('log')
    plt.hlines(min(expon_lr.losses), min(expon_lr.rates), max(expon_lr.rates))
    plt.axis([min(expon_lr.rates), max(expon_lr.rates), 0, expon_lr.losses[0]])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    plt.savefig('loss_lr.png')

#explore_properiate_lr()

# # Plot 
# import pandas as pd

# pd.DataFrame(history.history).plot(figsize=(8, 5))
# plt.grid(True)
# plt.gca().set_ylim(0, 1)
# plt.show()

def build_model2(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

def tuning():
    from scipy.stats import reciprocal
    from sklearn.model_selection import RandomizedSearchCV

    param_distribs = {
        "n_hidden": [0, 1, 2, 3],
        "n_neurons": np.arange(1, 100),
        "learning_rate": reciprocal(3e-4, 3e-2),
    }

    keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model2)

    rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3, verbose=2)
    rnd_search_cv.fit(X_train, y_train, epochs=100,
                    validation_data=(X_valid, y_valid),
                    callbacks=[keras.callbacks.EarlyStopping(patience=10)])

    print(rnd_search_cv.best_params_)
    rnd_search_cv.score(X_test, y_test)
    model3 = rnd_search_cv.best_estimator_.model
    print(model3.evaluate(X_test, y_test))


# tuning()
# model = keras.models.load_model("keras_seq_minst_model.h5") # rollback to best model
# print(model.evaluate(X_test, y_test))