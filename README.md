# Sentence-USE: Universal Sentence Encoder with Siamese Architecture

# Train and evaluate on STSb dataset

## Download STSb dataset

To download and prepare the STSb dataset, run the following script:

```
python sentence_use/data/stsb.py
```

It will download the train, dev and test datasets.

## Evaluate

To run the evaluation on the STSb test benchmark and calculate the pearson correlation and the spearman`s rank correlation to the groundtruth, one can run the following script:

```
python evaluate.py --eval-data=stsb_test.csv \
                    --model-name-or-path=https://tfhub.dev/google/universal-sentence-encoder/4
```

It should output the evaluation metrics:

```
Pearsons correlation: 0.7873, Spearman`s rank correlation: 0.7709,
```

## Run training

Expects a training and validation dataset. Dataset needs to be provided as csv file with `score`, `sentence1` and `sentence2` columns. Scores must be in range [0, 1].

```
python train.py --train-data=stsb_train.csv \
                --val-data=stsb_dev.csv \
                --model-name-or-path=https://tfhub.dev/google/universal-sentence-encoder/4 \
                --lr=0.0001 \
                --epochs=1 \
                --batch-size=8
```

The script will save the model under the path provided with the `output-path`. Per default the model is saved under `savedmodel`.

## Evaluate again

To run the evaluation again after the training, run the evaluate script again and provide the path to the trained model.

```
python evaluate.py --eval-data=stsb_test.csv \
                   --model-name-or-path=savedmodel
```

Output:

```
Pearsons correlation: 0.8042, Spearman`s rank correlation: 0.7915
```
