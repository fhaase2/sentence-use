import argparse

train_args = argparse.ArgumentParser(description="Training script for Siamese-USE")

train_args.add_argument(
    "--train-data",
    required=True,
    dest="train_data",
    action="store",
    help="Path to the input data with training data")

train_args.add_argument(
    "--val-data",
    required=True,
    dest="val_data",
    action="store",
    help="Path to the input data with validation data")

train_args.add_argument(
    "--model-name-or-path",
    dest="model_name_or_path",
    action="store",
    default="https://tfhub.dev/google/universal-sentence-encoder/4",
    help="Hub name or file path")


train_args.add_argument(
    "--output-path",
    dest="output_path",
    action="store",
    default="savedmodel",
    help="Path to store saved models and checkpoints")

train_args.add_argument(
    "--lr",
    dest="lr",
    action="store",
    default=0.001,
    type=float,
    help="Learning rate")

train_args.add_argument(
    "--epochs",
    dest="epochs",
    action="store",
    default=1,
    type=int,
    help="Number of training epochs")

train_args.add_argument(
    "--batch-size",
    dest="batch_size",
    action="store",
    default=8,
    type=int,
    help="Batch size")

train_args.add_argument(
    "--optimizers",
    dest="optimizer",
    action="store",
    default="adam",
    type=str,
    help="Optimizer")

train_args.add_argument(
    "--loss",
    dest="loss",
    action="store",
    default="mse",
    type=str,
    help="Loss")

train_args.add_argument(
    "--metric",
    dest="metric",
    action="store",
    default="mse",
    type=str,
    help="Loss")

train_args.add_argument(
    "--seed",
    dest="seed",
    action="store",
    default=42,
    help="Seed")


eval_args = argparse.ArgumentParser(description="Eval script for Siamese-USE")

eval_args.add_argument(
    "--eval-data",
    required=True,
    dest="eval_data",
    action="store",
    help="Path to the evaluation data")

eval_args.add_argument(
    "--model-name-or-path",
    dest="model_name_or_path",
    action="store",
    default="https://tfhub.dev/google/universal-sentence-encoder/4",
    help="Hub name or file path")

eval_args.add_argument(
    "--seed",
    dest="seed",
    action="store",
    default=42,
    help="Seed")
