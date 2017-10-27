import numpy as np
import pickle
from collections import Counter


def build_dictionary(filename, vocab_size, delimiter, position):
    word_counts = Counter()
    max_length = 0
    with open(filename, "r", encoding="utf-8") as source:
        for line in source:
            text = line.strip().split(delimiter)[position]
            words = text.split()
            word_counts.update(words)
            max_length = max(max_length, len(words))
    return {w[0]: i + 1 for i, w in enumerate(word_counts.most_common(vocab_size))}, max_length

class TextFileDataset(object):
    def __init__(self, delimiter="\t", position=3):
        self.position = position
        self.delimiter = delimiter

    def load_vocab(self, filename, vocab_size):
        self.word_to_index, self.input_length = build_dictionary(filename, vocab_size, self.delimiter, self.position)
        self.index_to_word = {w: i for i, w in self.word_to_index.items()}

    def save_vocab(self, filename):
        with open(filename, "wb") as target:
            pickle.dump((self.word_to_index, self.index_to_word, self.input_length), target)

    def reload_vocab(self, filename):
        with open(filename, "rb") as source:
            self.word_to_index, self.index_to_word, self.input_length = pickle.load(source)

    def vocab_size(self):
        return len(self.word_to_index) + 1

    def text(self, line):
        return " ".join(self.index_to_word.get(x, "<UNK>") for x in line)

    def close(self):
        if not self.source_file.closed:
            self.source_file.close()

    def batch(self, batch_size):
        result = []
        for i in range(batch_size):
            line = next(self.source)
            if line is None:
                break
            result.append(self.process_line(line))
        return np.array(result)

    def batches(self, batch_size, equal_batches=False):
        alive = True
        while alive:
            result = self.batch(batch_size)
            alive = len(result[0]) == batch_size
            if len(result[0]) == batch_size or (not equal_batches and len(result[0]) > 0):
                yield result
        self.reset()

    def process_line(self, line):
        line = [self.word_to_index.get(w, 0) for w in line]
        if len(line) < self.input_length:
            return line + [0] * (self.input_length - len(line))
        else:
            return line[: self.input_length]

    def get_source(self):
        with open(self.source_file, "r", encoding="utf-8") as source:
            for line in source:
                text = line.strip().split(self.delimiter)[self.position]
                words = text.split()
                yield words
            yield None

    def reset(self):
        self.source = self.get_source()

    def load(self, filename):
        self.source_file = filename
        self.reset()


class BowFileDataset(TextFileDataset):
    def __init__(self, stopword_file, delimiter="\t", position=3):
        super(BowFileDataset, self).__init__(delimiter=delimiter, position=position)
        with open(stopword_file, "r", encoding="utf-8") as source:
            self.stopwords = set([x.strip() for x in source])

    def load_vocab(self, filename, vocab_size, delimiter="\t"):
        self.word_to_index, _ = build_dictionary(filename, vocab_size, delimiter, self.position)
        self.index_to_word = {w: i for i, w in self.word_to_index.items()}
        self.input_length = 0
        with open(filename, "r", encoding="utf-8") as source:
            for line in source:
                text = line.strip().split(delimiter)[self.position].split()
                self.input_length = max(self.input_length,
                                        len(np.unique([self.word_to_index.get(w, 0)
                                                       for w in text
                                                       if w not in self.stopwords])))

    def process_line(self, line):
        line = np.unique([self.word_to_index.get(w, 0) for w in line if w not in self.stopwords]).tolist()
        if len(line) < self.input_length:
            return line + [0] * (self.input_length - len(line))
        else:
            return line[: self.input_length]

    def batch(self, batch_size):
        x = super(BowFileDataset, self).batch(batch_size)
        return x, x


if __name__ == "__main__":
    data = BowFileDataset(stopword_file="stopwords.txt")
    data.load_vocab("sample.txt", 400)
    data.load("sample.txt")
    print(data.input_length)
    for x in data.batches(2, True):
        print(x[0].shape, data.text(x[0][0]))