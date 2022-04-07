import time

import numpy as np
import tensorflow as tf
from tensorflow import keras as KER


class NeuralNet:
    def __init__(self, lrate=0.01, optimizer='SGD', loss='categorical_crossentropy', in_shape=(2,),
                 nn_dims=(1024, 512, 32, 1), hidden_act_func='relu', episodes_per_game=50,
                 checkpoint_path='models/cp-{epoch:04d}.ckpt', label=''):
        tf.get_logger().setLevel('INFO')
        self.model = self.gennet(lrate=lrate, optimizer=optimizer, loss=loss, in_shape=in_shape,
                                 nn_dims=nn_dims, hidden_act_func=hidden_act_func)
        self.episodes_per_game = episodes_per_game
        self.episode_count = 0
        self.checkpoint_path = checkpoint_path
        self.label = label
        self.lite_model = LiteModel.from_keras_model(self.model)

    def gennet(self, lrate, optimizer, loss, in_shape, nn_dims, hidden_act_func):
        optimizer = eval('KER.optimizers.' + optimizer)
        loss = eval('KER.losses.' + loss) if type(loss) == str else loss

        model = KER.Sequential()
        model.add(tf.keras.layers.Dense(nn_dims[0], activation=hidden_act_func, name=f'input_layer',
                  input_shape=in_shape))
        for i in range(1, len(nn_dims) - 1):
            model.add(tf.keras.layers.Dense(nn_dims[i], activation=hidden_act_func, name=f'layer{i}'))
        model.add(tf.keras.layers.Dense(nn_dims[-1], activation='softmax', name='output_layer'))

        model.compile(optimizer=optimizer(learning_rate=lrate), loss=loss, metrics=[KER.metrics.categorical_accuracy])
        return model

    def predict(self, state):
        # return self.model(state)
        return self.lite_model.predict(state)

    def fit(self, x, y, callbacks=None):
        if self.episode_count == 0: self.save()
        self.episode_count += 1
        self.model.fit(x=x, y=y, callbacks=callbacks)
        if self.episode_count % self.episodes_per_game == 0: self.save()
        start_time = time.time()
        self.lite_model = LiteModel.from_keras_model(self.model)
        end_time = time.time()
        print(f'Lite model construction time: {(end_time-start_time):.4f}')

    def save(self):
        print('Saving model...\n')
        self.model.save(self.checkpoint_path.format(episode=self.episode_count))

    def load(self, episode_count):
        # print('\nBefore: ')
        # print(self.model.layers[0].get_weights()[0])
        self.model = tf.keras.models.load_model(self.checkpoint_path.format(episode=episode_count))
        self.lite_model = LiteModel.from_keras_model(self.model)
        # print('After:')
        # print(self.model.layers[0].get_weights()[0])


class LiteModel:

    @classmethod
    def from_file(cls, model_path):
        return LiteModel(tf.lite.Interpreter(model_path=model_path))

    @classmethod
    def from_keras_model(cls, kmodel):
        converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
        tflite_model = converter.convert()
        return LiteModel(tf.lite.Interpreter(model_content=tflite_model))

    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        input_det = self.interpreter.get_input_details()[0]
        output_det = self.interpreter.get_output_details()[0]
        self.input_index = input_det["index"]
        self.output_index = output_det["index"]
        self.input_shape = input_det["shape"]
        self.output_shape = output_det["shape"]
        self.input_dtype = input_det["dtype"]
        self.output_dtype = output_det["dtype"]

    def predict(self, inp):
        inp = inp.astype(self.input_dtype)
        count = inp.shape[0]
        out = np.zeros((count, self.output_shape[1]), dtype=self.output_dtype)
        for i in range(count):
            self.interpreter.set_tensor(self.input_index, inp[i:i + 1])
            self.interpreter.invoke()
            out[i] = self.interpreter.get_tensor(self.output_index)[0]
        return out

    def predict_single(self, inp):
        """ Like predict(), but only for a single record. The input data can be a Python list. """
        inp = np.array([inp], dtype=self.input_dtype)
        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_index)
        return out[0]
