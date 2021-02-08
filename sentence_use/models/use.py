import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text


class USE(tf.keras.Model):

    def __init__(self,
                 model_name_or_path="https://tfhub.dev/google/universal-sentence-encoder/4",
                 trainable=False,
                 **kwargs):
        super(USE, self).__init__()
        self.model_name_or_path = model_name_or_path
        self.embed = hub.KerasLayer(handle=model_name_or_path,
                                    input_shape=[],
                                    dtype=tf.string,
                                    trainable=trainable,
                                    **kwargs)

    def call(self, inputs):
        return self.embed(inputs)
