import csv


def read_data(path):
    """Reads csv dataset and returns in the correct format.

    :param path: Path of file.
    :type path: str
    :rtype: tuple
    """
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        sentences = []
        labels = []
        for row in reader:
            score = float(row["score"])
            sentences.append((row["sentence1"], row["sentence2"]))
            labels.append(score)
        return sentences, labels
