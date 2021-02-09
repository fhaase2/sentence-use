import tensorflow as tf

from sentence_use.data import read_data
from sentence_use.models import SiameseUSE
from sentence_use.parser import train_args


def train(args):
    """Runs training script for given CLI arguments.

    :param args: Arguments
    :type args: argparse.Namespace
    """
    tf.random.set_seed(args.seed)

    # x_train and x_val format is [("sentence1", "sentence2")]
    # y_train and y_val format is [0.95]
    x_train, y_train = read_data(args.train_data)
    x_val, y_val = read_data(args.val_data)

    # Prepare train and val datasets as tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(len(x_train), seed=args.seed).batch(args.batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(args.batch_size)

    model = SiameseUSE(model_name_or_path=args.model_name_or_path,
                       trainable=True)

    model.compile(
        optimizer=args.optimizer,
        loss=args.loss,
        metrics=args.metric,
    )
    model.optimizer.learning_rate.assign(float(args.lr))

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir="./logs"),
    ]

    model.fit(
        train_dataset,
        epochs=args.epochs,
        validation_data=val_dataset,
        callbacks=callbacks
    )

    model.model.save(args.output_path)


if __name__ == "__main__":
    args = train_args.parse_args()
    train(args)
