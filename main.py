import dataset
import models


def test_model():
    embedding_dim = 100
    num_latent_factors = 50

    data = dataset.BowFileDataset(stopword_file="stopwords.txt")
    # data.load_vocab("vne_lower.txt", 10000)
    # print(data.vocab_size())
    # data.save_vocab("vne.10k.vocab")
    data.reload_vocab("vne.10k.vocab")
    data.load("vne_lower.txt")
    print(data.input_length)

    model = models.SigmoidVariationalBowModel(input_dim=data.vocab_size(),
                                              output_dim=data.vocab_size(),
                                              embedding_dim=embedding_dim,
                                              num_latent_factors=num_latent_factors)
    model.load("vne.2.model")
    # model.fit(data, batch_size=128, num_epochs=2, verbose=2)
    # model.save("vne.2.model")
    ranks = model.rank(data, batch_size=128, verbose=1)
    for i in range(20):
        print(data.text(ranks[i]))

if __name__ == "__main__":
    test_model()
