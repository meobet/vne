import dataset
import models
import pyro_test

from operations import numpy
import torch
from torch.autograd import Variable

def test_model():
    embedding_dim = 100
    num_latent_factors = 50

    data = dataset.BowFileDataset(stopword_file="stopwords.txt", position=7)
    # data.load_vocab("vne.txt", 10000)
    # print(data.vocab_size())
    # data.save_vocab("vne.10k.vocab")
    data.reload_vocab("vne.10k.vocab")
    data.load("vne.txt")
    print(data.input_length)

    model = pyro_test.SigmoidVariationalBowModel(input_dim=data.vocab_size(),
                                              output_dim=data.vocab_size(),
                                              embedding_dim=embedding_dim,
                                              num_latent_factors=num_latent_factors)
    # model.load("vne.5.model")
    model.fit(data, batch_size=128, num_epochs=5, verbose=2)
    model.save("pyro.vne.5.model")
    ranks, norms = model.sample_from_latent(20, 20)
    ranks = model.rank(data, batch_size=128, num_batches=1, verbose=1)
    # ranks = model.top_n(data, batch_size=128, num_batches=1, verbose=1, n=20)
    for i in range(20):
        print(data.text(ranks[i][::-1]))

if __name__ == "__main__":
    test_model()
