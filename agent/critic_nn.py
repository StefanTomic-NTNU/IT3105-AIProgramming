import numpy as np
import tensorflow as tf
from keras.optimizers import adam_v2
import tensorflow_probability as tfp

from agent.neural_network import NeuralNetwork


class CriticNN():
    def __init__(self, alpha=0.0003, gamma=0.99):
        self.gamma = gamma
        self.neural_network = NeuralNetwork()
        self.neural_network.compile(optimizer=adam_v2.Adam(learning_rate=alpha))
        self.__td_error = 0

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        _, probs = self.neural_network(state)

    def update_td_error(self, prev_state, new_state, reward, done):
        prev_state = tf.convert_to_tensor([prev_state], dtype=tf.float32)
        new_state = tf.convert_to_tensor([new_state], dtype=tf.float32)
        reward = tf.convert_to_tensor([reward], dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            prev_state_value = self.neural_network(prev_state)
            new_state_value = self.neural_network(new_state)
            prev_state_value = tf.squeeze(prev_state_value)
            new_state_value = tf.squeeze(new_state_value)

            # print(reward)
            # print(self.gamma)
            # print(new_state_value)
            # print(1 - int(done))
            # print(prev_state_value)
            delta = reward + self.gamma * new_state_value * (1 - int(done)) - prev_state_value
            loss = delta**2

            gradient = tape.gradient(loss, self.neural_network.trainable_variables)
            self.neural_network.optimizer.apply_gradients(zip(gradient, self.neural_network.trainable_variables))

            target = reward + self.gamma * self.stateValue(new_state)
            self.__td_error = tf.keras.backend.eval(target - self.stateValue(prev_state))[0]

    def stateValue(self, state):
        # state = [tf.float32.to_number(bin, out_type=tf.dtypes.int32) for bin in state]  # convert to array
        state = tf.convert_to_tensor(np.expand_dims(state, axis=0))
        return self.neural_network(state).numpy()[0][0]

    def get_delta(self):
        return self.__td_error

    def update_elig(self, prev_state):
        pass

    def init_eval(self, new_state):
        pass

    def update_evals(self):
        pass

    def decay_eligs(self):
        pass

    def get_td_error(self):
        return self.__td_error

    def new_episode(self):
        pass
            
