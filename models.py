import numpy as np
import math
import time
import torch
from torch.autograd import Variable
from torch.nn import Module, Parameter, Embedding, Linear, ReLU, Sigmoid, BCEWithLogitsLoss
from operations import variable, numpy, exp_lr_scheduler, to_binary


class SigmoidVariationalBowModel(Module):
    def __init__(self, input_dim, output_dim, embedding_dim, num_latent_factors, use_cuda=None):
        super(SigmoidVariationalBowModel, self).__init__()

        self.num_samples = 0
        self.num_latent_factors = num_latent_factors

        self.input_embedding = Embedding(num_embeddings=input_dim, embedding_dim=embedding_dim)
        self.input_embedding.weight.data.uniform_(-1. / math.sqrt(input_dim), 1. / math.sqrt(input_dim))
        self.output_embedding = Embedding(num_embeddings=output_dim, embedding_dim=num_latent_factors)

        self.input_bias = Parameter(torch.zeros(embedding_dim))
        self.output_bias = Parameter(torch.zeros(output_dim))

        self.mu = Linear(embedding_dim, num_latent_factors, bias=True)
        self.logvar = Linear(embedding_dim, num_latent_factors, bias=True)

        self.lr_scheduler = exp_lr_scheduler
        self.lr = 0.001
        self.optimizer = None

        self.relu = ReLU()
        self.label_loss = BCEWithLogitsLoss()
        self.sigmoid = Sigmoid()

        if use_cuda is None:
            use_cuda = torch.cuda.is_available()
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()

    def live_params(self):
        return [param for param in self.parameters() if param.requires_grad]

    def build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.live_params(), lr=self.lr)

    def cuda(self, device_id=None):
        super(SigmoidVariationalBowModel, self).cuda(device_id)
        self.use_cuda = True

    def encode(self, x):
        embedding = torch.bmm((x > 0).float().unsqueeze(dim=1), self.input_embedding(x)).squeeze(dim=1)
        embedding = self.relu(embedding + self.input_bias)
        return self.mu(embedding), self.logvar(embedding)

    def sample_z(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.use_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        logits = z.mm(self.output_embedding.weight.t())
        logits = logits + self.output_bias
        return logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample_z(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, x, y):
        y_hat, mu, logvar = self.forward(x)
        label_loss = self.label_loss(y_hat, Variable(to_binary(y.data, y_hat.size(), use_cuda=self.use_cuda)))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / float(np.product(x.size()))

        return label_loss, KLD

    def predict(self, x, num_samples=None):
        mu, logvar = self.encode(x)
        if num_samples is None:
            num_samples = self.num_samples
        if num_samples == 0:
            return self.decode(mu)
        else:
            samples = [self.decode(self.sample_z(mu, logvar)) for i in range(num_samples)]
            return torch.stack(samples).mean(dim=0)

    def fit(self, dataset, batch_size, num_epochs=1, verbose=0):
        self.train(True)
        if self.optimizer is None:
            self.build_optimizer()

        fit_loss = []
        for epoch in range(num_epochs):
            self.lr = self.lr_scheduler(self.optimizer, epoch)
            epoch_loss = []
            timer = time.time()

            # Iterate over data.
            for x, y in dataset.batches(batch_size):
                # get the inputs
                inputs, labels = variable(self, (torch.from_numpy(x).long(), torch.from_numpy(y).long()))

                # zero the parameter gradients
                self.optimizer.zero_grad()
                label_loss, KLD = self.loss(inputs, labels)
                loss = label_loss + KLD
                loss.backward()
                self.optimizer.step()

                # statistics
                if verbose > 1:
                    print("Batch",
                          len(epoch_loss),
                          "loss:",
                          numpy(self, loss),
                          "=",
                          numpy(self, label_loss),
                          "+",
                          numpy(self, KLD),
                          "average time:",
                          (time.time() - timer) / float(len(epoch_loss) + 1))
                epoch_loss.append((numpy(self, label_loss)[0], numpy(self, KLD)[0]))
            if verbose > 0:
                print("loss =",
                      np.mean(epoch_loss, axis=0),
                      "time =",
                      time.time() - timer)
            fit_loss.append(np.mean(epoch_loss))
        return fit_loss

    def top_n(self, dataset, n, batch_size, num_batches=None, verbose=0):
        self.train(False)
        timer = time.time()
        result = []
        batch_count = 0.
        # Iterate over data.
        for x, y in dataset.batches(batch_size=batch_size):
            # get the inputs
            inputs, labels = variable(self, (torch.from_numpy(x).long(), torch.from_numpy(y).long()))
            batch_count += 1
            if num_batches is not None and batch_count > num_batches:
                break
            if verbose > 0:
                print("Batch", batch_count, "average time:", (time.time() - timer) / batch_count)
            result.append(numpy(self, self.predict(inputs)).argsort(axis=1)[:, -1:-n-1:-1].copy())

        return np.vstack(result)

    def rank(self, dataset, batch_size, num_batches=None, verbose=0):
        self.train(False)
        timer = time.time()
        result = []
        batch_count = 0.
        # Iterate over data.
        for x, y in dataset.batches(batch_size=batch_size):
            # get the inputs
            inputs, labels = variable(self, (torch.from_numpy(x).long(), torch.from_numpy(y).long()))
            batch_count += 1
            if num_batches is not None and batch_count > num_batches:
                break
            if verbose > 0:
                print("Batch", batch_count, "average time:", (time.time() - timer) / batch_count)
            candidates = [x[x > 0] for x in numpy(self, inputs)]
            outputs = [sorted(zip(x, y[x]), key=lambda t: t[1])
                       for x, y in zip(candidates, numpy(self, self.predict(inputs)))]
            result.extend([[y[0] for y in x[::-1]] for x in outputs])

        return result

    def sample_from_latent(self, num_samples, top_n):
        self.train(False)
        latents = variable(self, torch.randn(num_samples, self.num_latent_factors))
        return numpy(self, self.decode(latents)).argsort(axis=1)[:, -1:-top_n-1:-1].copy(), \
               np.linalg.norm(numpy(self, latents), axis=1)

    def ml_estimate(self, dataset, batch_size, num_batches=None, verbose=0):
        self.train(False)

    def save(self, filename):
        torch.save({"state_dict": self.state_dict(), "lr": self.lr}, filename)

    def load(self, filename):
        data = torch.load(filename)
        self.load_state_dict(data["state_dict"])
        self.lr = data["lr"]