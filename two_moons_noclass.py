#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss

import numpy as np
import sklearn.datasets

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# In[ ]:


class Model_bilinear(nn.Module):
    def __init__(self, features, num_embeddings):
        super().__init__()

        self.gamma = 0.99
        self.sigma = 0.3

        #embedding_size = 10

        self.fc1 = nn.Linear(2, features)
        self.fc2 = nn.Linear(features, features)
        self.fc3 = nn.Linear(features, features)

        #self.W = nn.Parameter(torch.normal(torch.zeros(embedding_size, num_embeddings, features), 1))
        self.W = nn.Parameter(torch.ones(num_embeddings, 1))
        self.register_buffer('N', torch.ones(num_embeddings) * 20)
        self.register_buffer('m', torch.normal(torch.zeros(num_embeddings,features), 1))

        #self.m = self.m * self.N.unsqueeze(0)

    def embed(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # i is batch, m is embedding_size, n is num_embeddings (classes)
        #x = torch.einsum('ij,mnj->imn', x, self.W)

        return x

    def bilinear(self, z):
        #embeddings = self.m / self.N.unsqueeze(0)
        embeddings = self.m
        z_norm = (z** 2).sum(1).view(-1, 1)
        points_norm = (embeddings ** 2).sum(1).view(1, -1)
        d2 = z_norm + points_norm - 2.0 * torch.mm(z, embeddings.transpose(0, 1))
        #diff = z - embeddings.unsqueeze(0)
        #y_pred = (- diff ** 2).mean(1).div(2 * self.sigma ** 2).exp()
        y_dis = (- d2.clamp(min=0.0)).div(2 * self.sigma ** 2).exp()
        y_pred=torch.mm(y_dis,F.softplus(self.W))
        return y_pred

    def forward(self, x):
        z = self.embed(x)
        y_pred = self.bilinear(z)

        return z, y_pred

    def update_embeddings(self, x, y):
        z = self.embed(x)

        # normalizing value per class, assumes y is one_hot encoded
        self.N = torch.max(self.gamma * self.N + (1 - self.gamma) * y.sum(0), torch.ones_like(self.N))

        # compute sum of embeddings on class by class basis
        features_sum = torch.einsum('ijk,ik->jk', z, y)

        self.m = self.gamma * self.m + (1 - self.gamma) * features_sum


# In[ ]:


np.random.seed(0)
torch.manual_seed(0)

l_gradient_penalty = 1.0

# Moons
noise = 0.1
X_train, y_train = sklearn.datasets.make_moons(n_samples=1500, noise=noise)
X_test, y_test = sklearn.datasets.make_moons(n_samples=200, noise=noise)
plt.scatter(X_train[:,0],X_train[:,1])
plt.show()

num_classes = 2
batch_size = 64

model = Model_bilinear(20, num_classes)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)


def calc_gradient_penalty(x, y_pred):
    gradients = torch.autograd.grad(
        outputs=y_pred,
        inputs=x,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
    )[0]

    gradients = gradients.flatten(start_dim=1)

    # L2 norm
    grad_norm = gradients.norm(2, dim=1)

    # Two sided penalty
    gradient_penalty = ((grad_norm - 1) ** 2).mean()

    # One sided penalty - down
    #     gradient_penalty = F.relu(grad_norm - 1).mean()

    return gradient_penalty


def output_transform_acc(output):
    y_pred, y, x, z = output

    y = torch.argmax(y, dim=1)

    return y_pred, y


def output_transform_bce(output):
    y_pred, y, x, z = output

    return y_pred, y


def output_transform_gp(output):
    y_pred, y, x, z = output

    return x, y_pred


def step(engine, batch):
    model.train()
    optimizer.zero_grad()

    x, y = batch
    x.requires_grad_(True)

    z, y_pred = model(x)

    loss1 = F.binary_cross_entropy(y_pred, y)
    loss2 = l_gradient_penalty * calc_gradient_penalty(x, y_pred)

    loss = loss1 + loss2

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        model.update_embeddings(x, y)

    return loss.item()


def eval_step(engine, batch):
    model.eval()

    x, y = batch

    x.requires_grad_(True)

    z, y_pred = model(x)

    return y_pred, y, x, z


trainer = Engine(step)
evaluator = Engine(eval_step)

metric = Accuracy(output_transform=output_transform_acc)
metric.attach(evaluator, "accuracy")

metric = Loss(F.binary_cross_entropy, output_transform=output_transform_bce)
metric.attach(evaluator, "bce")

metric = Loss(calc_gradient_penalty, output_transform=output_transform_gp)
metric.attach(evaluator, "gp")

ds_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(),
                                          F.one_hot(torch.from_numpy(y_train)).float())
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)

ds_test = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), F.one_hot(torch.from_numpy(y_test)).float())
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=200, shuffle=False)


@trainer.on(Events.EPOCH_COMPLETED)
def log_results(trainer):
    evaluator.run(dl_test)
    metrics = evaluator.state.metrics

    print("Test Results - Epoch: {} Acc: {:.4f} BCE: {:.2f} GP {:.2f}"
          .format(trainer.state.epoch, metrics['accuracy'], metrics['bce'], metrics['gp']))


# In[ ]:


trainer.run(dl_train, max_epochs=30)

# In[ ]:


domain = 3
x_lin = np.linspace(-domain + 0.5, domain + 0.5, 100)
y_lin = np.linspace(-domain, domain, 100)

xx, yy = np.meshgrid(x_lin, y_lin)

X_grid = np.column_stack([xx.flatten(), yy.flatten()])

X_vis, y_vis = sklearn.datasets.make_moons(n_samples=1000, noise=noise)
mask = y_vis.astype(np.bool)

with torch.no_grad():
    output = model(torch.from_numpy(X_grid).float())[1]
    confidence = output.max(1)[0].numpy()

z = confidence.reshape(xx.shape)

plt.figure()
plt.contourf(x_lin, y_lin, z, cmap='cividis')

plt.scatter(X_vis[mask, 0], X_vis[mask, 1])
plt.scatter(X_vis[~mask, 0], X_vis[~mask, 1])
plt.show()

# In[ ]:




