import torch
import torch.nn.functional as F

def entropy(logit : torch.Tensor, eps=1e-5, **kwargs):

    prob = F.softmax(logit, dim=1)
    ent = -1.0 * torch.sum(torch.log(prob + eps) * prob, dim=1)

    return ent


def maxlogit(logit : torch.Tensor, probability = False, **kwargs):

    if probability:
        prob = F.softmax(logit, dim=1)
    else:
        prob = logit

    max_logit, _ = torch.max(prob, dim=1)
    max_logit = 1.0 - max_logit

    return max_logit


def energy(logit : torch.Tensor, **kwargs):

    unormalized_prob = torch.exp(logit)
    energy = 1.0 - torch.log(torch.sum(unormalized_prob, dim=1))

    return energy


def get_estimator(name : str):

    if name == "entropy":
        return entropy
    elif name == "maxlogit":
        return maxlogit
    elif name == "energy":
        return energy
    else:
        return None
