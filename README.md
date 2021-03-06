# Sentence-USE: Universal Sentence Encoder with Siamese Architecture

This repository contains scripts to finetune the USE for a given semantic textual similarity dataset.

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

Expects a training and validation dataset. Dataset needs to be provided as csv file with `score`, `sentence1` and `sentence2` columns. Scores must be in range `[0, 1]`.

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

# References

- Yang, Y., Cer, D., Ahmad, A., Guo, M., Law, J., Constant, N., ... & Kurzweil, R. (2019). Multilingual universal sentence encoder for semantic retrieval. arXiv preprint arXiv:1907.04307.
- Cer, D., Yang, Y., Kong, S. Y., Hua, N., Limtiaco, N., John, R. S., ... & Kurzweil, R. (2018). Universal sentence encoder. arXiv preprint arXiv:1803.11175.
- Cer, D., Diab, M., Agirre, E., Lopez-Gazpio, I., & Specia, L. (2017). Semeval-2017 task 1: Semantic textual similarity-multilingual and cross-lingual focused evaluation. arXiv preprint arXiv:1708.00055.
