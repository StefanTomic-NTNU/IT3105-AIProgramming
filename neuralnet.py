import tensorflow as tf
from tensorflow import keras as KER


class NeuralNet:
    def __init__(self, lrate=0.01, optimizer='SGD', loss='categorical_crossentropy', in_shape=(2,),
                 nn_dims=(1024, 512, 32, 1), hidden_act_func='relu', M=10):
        self.model = self.gennet(lrate=lrate, optimizer=optimizer, loss=loss, in_shape=in_shape,
                                 nn_dims=nn_dims, hidden_act_func=hidden_act_func, M=M)

    def gennet(self, lrate, optimizer, loss, in_shape, nn_dims, hidden_act_func, M):
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
        return self.model(state)

    def fit(self, x, y, callbacks=None):
        self.model.fit(x=x, y=y, callbacks=callbacks)

    def save(self, path):
        pass
