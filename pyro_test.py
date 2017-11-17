import numpy as np
import math
import time
import torch
from torch.autograd import Variable
from torch.nn import Module, Parameter, Embedding, Linear, ReLU, Softplus, Sigmoid, BCEWithLogitsLoss
from operations import variable, numpy, exp_lr_scheduler, to_binary

import pyro
from pyro.infer import SVI, Importance, Marginal
from pyro.optim import Adam
from pyro.util import ng_zeros, ng_ones
import pyro.distributions as dist

fudge = 1e-7

class SigmoidVariationalBowModel(Module):
    def __init__(self, input_dim, output_dim, embedding_dim, num_latent_factors, use_cuda=None):
        super(SigmoidVariationalBowModel, self).__init__()

        self.num_samples = 0
        self.input_dim = input_dim
        self.num_latent_factors = num_latent_factors

        self.input_embedding = Linear(input_dim, embedding_dim)
        self.output_embedding = Linear(num_latent_factors, output_dim)

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
        embedding = self.relu(self.input_embedding(x))
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
        return self.output_embedding(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample_z(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, x, y):
        y_hat, mu, logvar = self.forward(x)
        label_loss = self.label_loss(y_hat, Variable(to_binary(y.data, y_hat.size(), use_cuda=self.use_cuda)))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / float(np.product(x.size()))

        return label_loss, KLD

    def pyro_model(self, x):
        pyro.module("output_embedding", self.output_embedding)
        mu = variable(self, torch.zeros(x.size(0), self.num_latent_factors))
        sigma = variable(self, torch.ones(x.size(0), self.num_latent_factors))
        z = pyro.sample("latent", dist.normal, mu, sigma)
        out_prob = (self.sigmoid(self.decode(z)) + fudge) * (1 - 2 * fudge)
        pyro.observe("obs", dist.bernoulli, x, out_prob)
        return z

    def pyro_guide(self, x):
        pyro.module("input_embedding", self.input_embedding)
        pyro.module("mu", self.mu)
        pyro.module("logvar", self.logvar)
        mu, logvar = self.encode(x)
        z = pyro.sample("latent", dist.normal, mu, torch.exp(logvar))
        return z

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
        svi = SVI(self.pyro_model, self.pyro_guide, Adam({"lr": self.lr}), loss="ELBO")

        fit_loss = []
        for epoch in range(num_epochs):
            epoch_loss = []
            timer = time.time()

            # Iterate over data.
            for x, y in dataset.batches(batch_size):
                x = to_binary(torch.from_numpy(x).long(), (x.shape[0], self.input_dim), use_cuda=self.use_cuda)
                inputs = variable(self, x)
                loss = svi.step(inputs)

                # statistics
                if verbose > 1:
                    print("Batch",
                          len(epoch_loss),
                          "loss:",
                          loss,
                          "average time:",
                          (time.time() - timer) / float(len(epoch_loss) + 1))
                epoch_loss.append(loss)
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
            x = to_binary(torch.from_numpy(x).long(), (x.size(0), self.input_dim), use_cuda=self.use_cuda)
            inputs = variable(self, x)
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
            x = to_binary(torch.from_numpy(x).long(), (x.size(0), self.input_dim), use_cuda=self.use_cuda)
            inputs = variable(self, x)
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