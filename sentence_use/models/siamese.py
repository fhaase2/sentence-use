import tensorflow as tf
import tensorflow_hub as hub

from sentence_use.models import USE


class SiameseUSE(tf.keras.Model):

    def __init__(self,
                 model_name_or_path="https://tfhub.dev/google/universal-sentence-encoder/4",
                 trainable=False,
                 **kwargs):
        super(SiameseUSE, self).__init__()
        self.model = USE(model_name_or_path=model_name_or_path,
                         trainable=trainable)
        self.cosine = tf.keras.layers.Dot(1, normalize=True)

    def call(self, inputs, training):
        sents_left = inputs[:, 0]
        sents_right = inputs[:, 1]
        embed_left = self.model(sents_left)
        embed_right = self.model(sents_right)
        cosine = self.cosine([embed_left, embed_right])

        return cosine
