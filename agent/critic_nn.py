import os

import tensorflow as tf
from keras.optimizers import adam_v2
import numpy as np

from agent.critic import Critic


class CriticNN(Critic):
    def __init__(self, layers, learning_rate=0.003, discount_factor=0.91,
                 name='actor_critic',
                 chkpt_dir='tmp/actor_critic'):
        super().__init__()
        self.layers = layers
        self.__learning_rate = learning_rate        # alpha
        self.__discount_factor = discount_factor    # gamma
        self.__td_error = 0                         # delta

        # variables used for checkpointing. Currently not in use.
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ac')

        # Compile model:
        model = tf.keras.Sequential()
        for i in range(0, len(layers)):
            if i == len(layers)-1:
                model.add(tf.keras.layers.Dense(layers[i]))
            else:
                model.add(tf.keras.layers.Dense(layers[i], activation="relu"))
        model.compile(optimizer=adam_v2.Adam(learning_rate=self.__learning_rate), run_eagerly=False)
        self.__eval_model = model

    def update_td_error(self, prev_state, new_state, reward, done):
        """
        Updates td error of critic
        :param prev_state:  s
        :param new_state:   s'
        :param reward:      reward from state transition
        :param done:        done flag
        """
        np_prev_state = tuple_to_np_array(prev_state)
        np_new_state = tuple_to_np_array(new_state)
        prev_state = tf.convert_to_tensor(np_prev_state)
        new_state = tf.convert_to_tensor(np_new_state)
        with tf.GradientTape() as tape:
            loss, td_error_tensor = get_loss(
                reward +
                self.__discount_factor *
                self.__eval_model(new_state),
                self.__eval_model(prev_state)
            )
        gradients = tape.gradient(loss, self.__eval_model.trainable_variables)
        self.__eval_model.optimizer.apply_gradients(zip(gradients, self.__eval_model.trainable_variables))
        self.__td_error = tf.keras.backend.eval(td_error_tensor)[0][0]

    def get_delta(self):
        return self.__td_error

    def get_td_error(self):
        return self.__td_error


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


def tuple_to_np_array(tup): return np.array(np.asarray(tup).flatten().reshape(1, -1))

