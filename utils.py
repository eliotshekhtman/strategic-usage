import torch
import numpy as np
from tqdm import tqdm

def memory(alphas, p=0.5):
    v = []
    m = []
    for i, a in enumerate(alphas):
        m.append(p ** (len(alphas) - i - 1))
        v.append(m[-1] * a)
    v = torch.sum(torch.stack(v), dim=0) / sum(m)
    return v

def cache_memory(alpha, mem, p=0.5):
    return (alpha + p * mem) / (1 + p)

def alpha_loss(alpha, y_hats, q=2.0):
    one_m = torch.ones(y_hats.shape[1])
    loss = torch.sum(torch.pow(alpha @ one_m, q)) / q - \
        torch.trace(alpha @ y_hats.T)
    return loss

def opt_alpha(y_hats, max_iters=1_000, lr=0.0001, q=2.0, quiet=True, round_lr=False): # n x m
    alpha = torch.zeros_like(y_hats)
    lrs = np.exp(np.linspace(np.log(0.1), np.log(lr), max_iters))
    for i in (pbar := tqdm(range(max_iters), position=0, leave=True, disable=quiet)):
        alpha = torch.maximum(torch.zeros_like(alpha), alpha).detach()
        alpha.requires_grad_(True)
        optimizer = torch.optim.Adam([alpha], lr=lrs[i])
        optimizer.zero_grad()
        loss = alpha_loss(alpha, y_hats, q=q)
        pbar.set_description(f'Loss: {loss:.2f}/Max: {torch.max(alpha):.2f}')
        loss.backward()
        optimizer.step()
    if round_lr:
        alpha = torch.round(alpha, decimals=int(1 - np.log10(lr)))
    alpha = torch.maximum(torch.zeros_like(alpha), alpha).detach()
    return alpha

def cap_utilities(y_hats):
    return np.minimum(y_hats, np.ones_like(y_hats))



def linear_clf(X, theta):
    return X @ theta.T

def maxlin_clf(X, theta):
    utility = X @ theta.T
    return torch.minimum(utility, torch.ones_like(utility))

def hinge_loss(y, y_hats, sample_weight):
    losses = sample_weight * torch.maximum(torch.zeros_like(y_hats), 1 - y * y_hats)
    return torch.sum(losses)

def linear_loss(y, y_hats, sample_weight):
    losses = - sample_weight * y * y_hats
    return torch.sum(losses)



def str_var(name, v):
    return r' $' + repr(name)[1:-1] + r'=' + repr(str(v))[1:-1] + r'$,'