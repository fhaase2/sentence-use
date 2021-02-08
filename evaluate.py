import logging

import tensorflow as tf
from sklearn.metrics.pairwise import paired_cosine_distances
from scipy.stats import pearsonr, spearmanr

from sentence_use.parser import eval_args
from sentence_use.models import SiameseUSE
from sentence_use.data import read_data


def evaluate(args):
    tf.random.set_seed(args.seed)

    x_test, y_test = read_data(args.eval_data)

    model = SiameseUSE(model_name_or_path=args.model_name_or_path,
                       trainable=False).model

    sents_left, sents_right = zip(*x_test)
    embeddings_left = model(sents_left).numpy()
    embeddings_right = model(sents_right).numpy()

    cosine_similarity = 1 - (paired_cosine_distances(embeddings_left, embeddings_right))
    pearson_correlation, _ = pearsonr(y_test, cosine_similarity)
    pearson_correlation = float(pearson_correlation)
    spearman_correlation, _ = spearmanr(y_test, cosine_similarity)
    spearman_correlation = float(spearman_correlation)

    logging.info(
        f"Pearsons correlation: {pearson_correlation:.4f}, "
        f"Spearman`s rank correlation: {spearman_correlation:.4f}, "
    )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    args = eval_args.parse_args()
    evaluate(args)
