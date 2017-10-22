import dataset
import models


def test_model():
    embedding_dim = 100
    num_latent_factors = 20

    data = dataset.BowFileDataset(stopword_file="stopwords.txt")
    data.load_vocab("10k.txt", 1000)
    data.load_vne("10k.txt")
    print(data.input_length)

    model = models.SigmoidVariationalBowModel(input_dim=data.vocab_size(),
                                              output_dim=data.vocab_size(),
                                              embedding_dim=embedding_dim,
                                              num_latent_factors=num_latent_factors)
    model.fit(data, batch_size=16)
    model.save("vne.model")


if __name__ == "__main__":
    test_model()

