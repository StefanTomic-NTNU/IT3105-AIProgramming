import tensorflow as tf
import numpy as np

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], "GPU")
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != "GPU"
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


class ANNCritic:
    def __init__(
        self,
        layers,
        state_size,
        learning_rate=0.001,
        discount_factor=0.9,
    ):
        self.layers = layers
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.v_model = self.build_model(self.layers, state_size)
        self.td_error = 0.0

    @tf.function
    def compute_loss(self, y_true, y_pred):
        td_error_tensor = y_true - y_pred
        squared_difference = tf.math.square(td_error_tensor)
        return squared_difference, td_error_tensor

    def update_td_error(self, curr_state, next_state, reward, done):
        s = tf.convert_to_tensor(to_np_array(curr_state))
        s_next = tf.convert_to_tensor(to_np_array(next_state))
        v_s_next = self.v_model(s_next)
        with tf.GradientTape() as tape:
            loss, td_error_tensor = self.compute_loss(
                reward + self.discount_factor * v_s_next, self.v_model(s)
            )
        grads = tape.gradient(loss, self.v_model.trainable_variables)
        self.v_model.optimizer.apply_gradients(
            zip(grads, self.v_model.trainable_variables)
        )
        self.td_error = tf.keras.backend.eval(td_error_tensor)[0][0]
        return self.td_error

    def build_model(self, layers, state_size):
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Dense(
                layers[0],
                # input_shape=(state_size,),
                activation="relu",
            )
        )
        for i in range(1, len(layers) - 2):
            model.add(tf.keras.layers.Dense(layers[i], activation="relu"))
        model.add(tf.keras.layers.Dense(layers[-1]))
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, run_eagerly=False)
        return model

    def get_delta(self):
        return self.td_error

    def update_elig(self, prev_state):
        pass

    def init_eval(self, new_state):
        pass

    def update_evals(self):
        pass

    def decay_eligs(self):
        pass

    def get_td_error(self):
        return self.td_error

    def new_episode(self):
        pass


def to_np_array(t):
    return np.array(np.asarray(t).flatten().reshape(1, -1))


def to_tuple(d):
    items = d.values()
    return tuple(items)

