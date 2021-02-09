import codecs
import csv
import os
import pathlib
import tarfile
import urllib.request


def download_stsb(dataset_path=""):
    """Downloads STSb dataset and saves as train, dev and
    test datasets.

    :param dataset_path: Output path, defaults to ""
    :type dataset_path: str, optional
    """
    stsb_dataset_path = os.path.join(dataset_path, "stsb.tar.gz")
    csv_fieldnames = ["sect", "dataset", "year", "id", "score", "sentence1", "sentence2"]
    url = "http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz"
    urllib.request.urlretrieve(url, stsb_dataset_path)

    train_samples = []
    dev_samples = []
    test_samples = []

    with tarfile.open(stsb_dataset_path, "r:gz", encoding="utf-8") as tar:
        for split in ["train", "dev", "test"]:
            with tar.extractfile(f"stsbenchmark/sts-{split}.csv") as f:
                ft = codecs.getreader("utf-8")(f)
                reader = csv.DictReader(ft,
                                        delimiter="\t",
                                        quoting=csv.QUOTE_NONE,
                                        fieldnames=csv_fieldnames)
                for row in reader:
                    # Normalize scores to range [0,1]
                    score = float(row['score']) / 5.0
                    sample = {
                        "sentence1": row["sentence1"],
                        "sentence2": row["sentence2"],
                        "score": score
                    }

                    if split == "train":
                        train_samples.append(sample)
                    elif split == "dev":
                        dev_samples.append(sample)
                    elif split == "test":
                        test_samples.append(sample)

    keys = ["sentence1", "sentence2", "score"]

    output_path = os.path.join(dataset_path, "stsb_train.csv")
    with open(output_path, "w", newline="") as out:
        dict_writer = csv.DictWriter(out, keys)
        dict_writer.writeheader()
        dict_writer.writerows(train_samples)

    output_path = os.path.join(dataset_path, "stsb_dev.csv")
    with open(output_path, "w", newline="") as out:
        dict_writer = csv.DictWriter(out, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dev_samples)

    output_path = os.path.join(dataset_path, "stsb_test.csv")
    with open(output_path, "w", newline="") as out:
        dict_writer = csv.DictWriter(out, keys)
        dict_writer.writeheader()
        dict_writer.writerows(test_samples)


if __name__ == "__main__":
    download_stsb()
