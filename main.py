import dataset
import models
import pyro_test

from operations import numpy
import torch
from torch.autograd import Variable

def test_model():
    embedding_dim = 100
    num_latent_factors = 50

    data = dataset.BowFileDataset(stopword_file="stopwords.txt", position=3)
    # data.load_vocab("vne.txt", 10000)
    # print(data.vocab_size())
    # data.save_vocab("vne.10k.vocab")
    data.reload_vocab("vne.10k.vocab")
    data.load("vne.txt.test")
    print(data.input_length)

    model = pyro_test.SigmoidVariationalBowModel(input_dim=data.vocab_size(),
                                              output_dim=data.vocab_size(),
                                              embedding_dim=embedding_dim,
                                              num_latent_factors=num_latent_factors)
    # model.load("train.5.model")
    model.fit_direct(data, batch_size=512, num_epochs=10, verbose=2)
    model.save("direct.10.model")
    top_n, norms = model.sample_from_latent(20, 20)
    x, y = data.batch(20)
    # top_n = model.top_n(x, y, n=20)
    top_n_rv = model.top_n_rv(x, y, n=20)
    ranks = [r[:20] for r in model.rank(x, y)]
    for i in range(20):
        print(data.text(top_n[i]))
        print(data.text(top_n_rv[i]))
        print(data.text(ranks[i]))
        print()

if __name__ == "__main__":
    test_model()
