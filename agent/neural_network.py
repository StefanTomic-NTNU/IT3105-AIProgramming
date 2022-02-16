import agent.critic
import os
import keras as keras
from keras.layers import Dense


class NeuralNetwork(keras.Model):
    def __init__(self,
                 fc1_dims=1024,
                 fc2_dims=512,
                 name='actor_critic',
                 chkpt_dir='tmp/actor_critic'):
        super(NeuralNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ac')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.v = Dense(1, activation=None)  # Eval

    def call(self, inputs, training=None, mask=None):
        # inputs = state
        value = self.fc1(inputs)
        value = self.fc2(value)

        v = self.v(value)

        return v

