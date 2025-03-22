
"""
Continuous authentication using device behavior patterns
"""
import tensorflow as tf

class BioAuth(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.lstm = tf.keras.layers.LSTM(64)
        self.attention = tf.keras.layers.Attention()
        
    def call(self, inputs):
        x = self.lstm(inputs)
        return self.attention([x, x])