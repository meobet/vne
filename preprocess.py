from pyvi.pyvi import ViTokenizer
from string import punctuation


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


if __name__ == "__main__":
    # normalize("vne.txt", "vne_norm.txt")
    # verify("vne_norm.txt", "\t")
    # tokenize("sample.txt", "sample.txt")
    # replace_numbers("vne_token.txt", "vne_tokenized.txt")
    lower_all("vne_tokenized.txt", "vne_lower.txt")