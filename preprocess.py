from pyvi.pyvi import ViTokenizer
from string import punctuation
from sklearn.model_selection import train_test_split

def is_punctuations(token):
    return all(x in punctuation for x in token)


def is_number(token):
    try:
        float(token)
    except ValueError:
        return False
    return True


def normalize(filename, outname):
    with open(filename, "r", encoding="utf-8") as source, open(outname, "w", encoding="utf-8") as target:
        num_tokens = None
        text = ""
        for i, line in enumerate(source):
            print(i)
            text = text + line
            count = text.count("##")
            if num_tokens is None:
                num_tokens = count
            elif num_tokens > count:
                text = text.strip() + " "
                continue
            elif num_tokens < count:
                print(i, "expected", num_tokens, "found", count)
                break
            result = text.replace("\t", " ").replace("##", "\t").replace("\\", "")
            target.write(result)
            text = ""


def verify(filename, delimiter):
    with open(filename, "r", encoding="utf-8") as source:
        num_tokens = None
        for i, line in enumerate(source):
            if i != 8121:
                continue
            count = line.count(delimiter)
            print(count)
            print(line)
            break
            if num_tokens is None:
                num_tokens = count
            if num_tokens != count:
                print(line)
                print("expected", num_tokens, "found", count)
                break


def tokenize(filename, outname, delimiter="\t"):
    with open(filename, "r", encoding="utf-8") as source, open(outname, "w", encoding="utf-8") as target:
        for i, line in enumerate(source):
            print(i)
            tokens = line.strip().split(delimiter)
            for j in [1, 3, 7, 8]:
                tokens[j] = ViTokenizer.tokenize(tokens[j])
            target.write(delimiter.join(tokens) + "\n")


def replace_numbers(filename, outname, delimiter="\t"):
    with open(filename, "r", encoding="utf-8") as source, open(outname, "w", encoding="utf-8") as target:
        for i, line in enumerate(source):
            print(i)
            tokens = line.strip().split(delimiter)
            for j in [1, 3, 7, 8]:
                tokens[j] = " ".join(["<NUMBER>" if is_number(x) else x for x in tokens[j].split()])
            target.write(delimiter.join(tokens) + "\n")

def lower_all(filename, outname, delimiter="\t"):
    with open(filename, "r", encoding="utf-8") as source, open(outname, "w", encoding="utf-8") as target:
        for i, line in enumerate(source):
            print(i)
            tokens = line.strip().split(delimiter)
            for j in [1, 3, 7, 8]:
                tokens[j] = tokens[j].lower()
            target.write(delimiter.join(tokens) + "\n")


def remove_links(filename, outname, delimiter="\t"):
    with open(filename, "r", encoding="utf-8") as source, open(outname, "w", encoding="utf-8") as target:
        for i, line in enumerate(source):
            print(i)
            tokens = line.strip().split(delimiter)
            for j in [3]:
                idx = tokens[j].find(" >")
                if idx >= 0:
                    tokens[j] = tokens[j][:idx]
            target.write(delimiter.join(tokens) + "\n")


def split_file(filename, test_size=0.1):
    with open(filename, "r", encoding="utf-8") as source:
        lines = source.readlines()
    x, y = train_test_split(lines, test_size=test_size)
    with open(filename + ".train", "w", encoding="utf-8") as target:
        target.writelines(x)
    with open(filename + ".test", "w", encoding="utf-8") as target:
        target.writelines(y)


if __name__ == "__main__":
    # normalize("vne.txt", "vne_norm.txt")
    # verify("vne_norm.txt", "\t")
    # tokenize("sample.txt", "sample.txt")
    # replace_numbers("vne_token.txt", "vne_tokenized.txt")
    # lower_all("vne_tokenized.txt", "vne_lower.txt")
    # remove_links("vne_lower.txt", "vne.txt")
    split_file("vne.txt")