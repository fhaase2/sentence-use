import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text


class USE(tf.keras.Model):

    def __init__(self,
                 model_name_or_path="https://tfhub.dev/google/universal-sentence-encoder/4",
                 trainable=False,
                 **kwargs):
        """Initializes a SiameseUSE model.

        :param model_name_or_path: Model name or path, defaults to "https://tfhub.dev/google/universal-sentence-encoder/4"
        :type model_name_or_path: str, optional
        :param trainable: If model should be loaded trainable, defaults to False
        :type trainable: bool, optional
        """
        super(USE, self).__init__()
        self.model_name_or_path = model_name_or_path
        self.embed = hub.KerasLayer(handle=model_name_or_path,
                                    input_shape=[],
                                    dtype=tf.string,
                                    trainable=trainable,
                                    **kwargs)

    def call(self, inputs):
        """Forward pass.

        :param inputs: Input tensors.
        :type inputs: tf.Tensor
        :rtype: tf.Tensor
        """
        return self.embed(inputs)
