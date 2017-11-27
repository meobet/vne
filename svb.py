import numpy as np
import math
import time
import torch
from torch.autograd import Variable
from torch.nn import Module, Linear, ReLU, Softplus, Sigmoid, BCEWithLogitsLoss
from operations import variable, numpy, exp_lr_scheduler, to_binary

import pyro
from pyro.infer import SVI, Importance, Marginal
from pyro.optim import Adam
import pyro.distributions as dist

fudge = 1e-7

class SigmoidVariationalBowModel(Module):
    def __init__(self, input_dim, embedding_dim, num_latent_factors, use_cuda=None):
        super(SigmoidVariationalBowModel, self).__init__()

        self.num_samples = 0
        self.input_dim = input_dim
        self.num_latent_factors = num_latent_factors

        self.input_embedding = Linear(input_dim, embedding_dim, bias=True)
        self.output_embedding = Linear(num_latent_factors, embedding_dim, bias=True)

        self.mu = Linear(embedding_dim, num_latent_factors, bias=True)
        self.logvar = Linear(embedding_dim, num_latent_factors, bias=True)
        self.logits = Linear(embedding_dim, input_dim, bias=True)

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
        return self.logits(self.relu(self.output_embedding(z)))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample_z(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, x):
        x_hat, mu, logvar = self.forward(x)
        label_loss = self.label_loss(x_hat, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (x.size(0) * self.input_dim)

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

    def posterior_latent(self, x, num_traces=100, num_samples=100):
        posterior = Importance(self.pyro_model, self.pyro_guide, num_traces)
        marginal = Marginal(posterior)
        return torch.stack([marginal(x) for _ in range(num_samples)])

    def variational_latent(self, x, num_traces=100, num_samples=100):
        z_mu, z_sigma = self.encode(x)
        return torch.stack([pyro.sample("latent", dist.normal, z_mu, torch.exp(z_sigma)) for _ in range(num_samples)])

    def predict_from_posterior(self, x, num_traces=100, num_samples=100, z_mean=True):
        z_sample = self.variational_latent(x, num_traces, num_samples)
        if z_mean:
            return self.decode(z_sample.mean(dim=0))
        else:
            return self.decode(z_sample).mean(dim=0)

    def posterior_rv(self, x, num_traces=100, num_samples=100):
        z_sample = self.variational_latent(x, num_traces, num_samples)
        out_prob = (self.sigmoid(self.decode(z_sample)) + fudge) * (1 - 2 * fudge)
        return 1.0 / (1.0 / out_prob).mean(dim=0)

    def fit_direct(self, dataset, batch_size, num_epochs=1, verbose=0):
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
                inputs = to_binary(torch.from_numpy(x).long(), (x.shape[0], self.input_dim), use_cuda=self.use_cuda)
                inputs = variable(self, inputs)
                labels = variable(self, torch.from_numpy(y).long())

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

    def fit(self, dataset, batch_size, num_epochs=1, verbose=0):
        self.train(True)
        svi = SVI(self.pyro_model, self.pyro_guide, Adam({"lr": self.lr}), loss="ELBO")

        fit_loss = []
        for epoch in range(num_epochs):
            epoch_loss = []
            timer = time.time()

            # Iterate over data.
            for x, y in dataset.batches(batch_size):
                inputs = to_binary(torch.from_numpy(x).long(), (x.shape[0], self.input_dim), use_cuda=self.use_cuda)
                inputs = variable(self, inputs)
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

    def top_n(self, x, y, n):
        inputs = to_binary(torch.from_numpy(x).long(), (x.shape[0], self.input_dim), use_cuda=self.use_cuda)
        inputs = variable(self, inputs)
        return numpy(self, self.predict_from_posterior(inputs, z_mean=False)).argsort(axis=1)[:, -1:-n-1:-1]

    def top_n_rv(self, x, y, n):
        inputs = to_binary(torch.from_numpy(x).long(), (x.shape[0], self.input_dim), use_cuda=self.use_cuda)
        inputs = variable(self, inputs)
        return numpy(self, self.posterior_rv(inputs)).argsort(axis=1)[:, -1:-n-1:-1]

    def rank(self, x, y):
        inputs = to_binary(torch.from_numpy(x).long(), (x.shape[0], self.input_dim), use_cuda=self.use_cuda)
        inputs = variable(self, inputs)
        candidates = [t[t > 0].astype(int) for t in x]
        outputs = [sorted(zip(x, y[x]), key=lambda t: t[1])
                   for x, y in zip(candidates, numpy(self, self.predict(inputs)))]
        return [[y[0] for y in x[::-1]] for x in outputs]

    def iterate(self, function, dataset, batch_size, num_batches=None, verbose=0, **kwargs):
        self.train(False)
        timer = time.time()
        result = []
        batch_count = 0.
        # Iterate over data.
        for x, y in dataset.batches(batch_size=batch_size):
            result.append(function(x, y, **kwargs).copy())
            batch_count += 1
            if num_batches is not None and batch_count > num_batches:
                break
            if verbose > 0:
                print("Batch", batch_count, "average time:", (time.time() - timer) / batch_count)
        return np.vstack(result)

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