import tensorflow as tf
import numpy as np
from tensorflow import keras

'''The critic holds the value function V(s)'''


class Critic_NN:
    def __init__(self, config: dict):
        self.learningRate = config['lr']
        self.discountfactor = config['discount_fact']
        self.eligibilitydecayrate = config['elig_decay_rate']
        self.inputDimension = config['input_dim']
        self.dimensions = config['dims']
        self.activation = config['activation']
        self.optimizer = config['optimizer']
        self.model = self.buildModel()

    def buildModel(self):
        model = tf.keras.Sequential()
        inputLayer = keras.layers.Dense(self.inputDimension, activation=self.activation, input_dim=self.inputDimension,
                                        name='hiddenLayer0')
        model.add(inputLayer)

        for i, dimension in enumerate(self.dimensions):
            model.add(
                keras.layers.Dense(dimension, activation=self.activation, name=f'hiddenLayer{i + 1}')
            )

        outputLayer = keras.layers.Dense(1, name="outputLayer")

        model.add(outputLayer)
        optimizer = keras.optimizers.Adagrad(learning_rate=self.learningRate)
        model.compile(optimizer=self.optimizer)
        model.summary()
        return model

    def get_TD_error(self, state, action, nextState, nextAction, reward):
        # print(f'State: {state}')
        sap = np.append(state, action)
        sap_next = np.append(nextState, nextAction)
        V = self.model(np.array(sap).reshape(1, -1))
        V_next = self.model(np.array(sap_next).reshape(1, -1))
        print(f'V: {V}, \t V_next: {V_next}')
        return (reward + V_next * self.discountfactor - V).numpy()[0][0]

    def customFit(self, state, action, tdError):
        sap = np.append(state, action)
        params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            V = self.model(np.array(sap).reshape(1, -1))
            gradient = tape.gradient(V, params)
            self.model.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
        return

    def custom_fit_2(self, state, new_state, reward):
        with tf.GradientTape() as tape:
            loss, td_error_tensor = get_loss(
                reward +
                self.discountfactor *
                self.model(np.array(new_state).reshape(1, -1)),
                self.model(np.array(state).reshape(1, -1))
            )
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return tf.keras.backend.eval(td_error_tensor)[0][0]

    def save_model(self, path: str):
        self.model.save(path)

    def load_model(self, path: str):
        self.model = tf.keras.models.load_model(path)


@tf.function
def get_loss(true_target, predicted_target):
    """
    Util func for calculaiting loss
    :param true_target:
    :param predicted_target:
    :return:
    """
    td_error_tensor = true_target - predicted_target
    loss = td_error_tensor**2
    return loss, td_error_tensor
