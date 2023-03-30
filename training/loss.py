# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""
from random import randint

import torch
from torch_utils import persistence

def heun_update(net, x_cur, t_cur, t_next, class_labels=None):
    # Euler step.
    denoised = net(x_cur, t_cur, class_labels).to(torch.float64)
    d_cur = (x_cur - denoised) / t_cur
    x_next = x_cur + (t_next - t_cur) * d_cur

    # Second order correction.
    denoised_next = net(x_next, t_next, class_labels).to(torch.float64)
    d_prime = (x_next - denoised_next) / t_next

    return 0.5 * (d_cur + d_prime)

#----------------------------------------------------------------------------
# Loss function corresponding to the consistency distillation objective in
# "Consistency models".
@persistence.persistent_class
class CDLoss:

    def __init__(self, N=18, rho=7., eps=0.002, t_max=80, solver=heun_update, dist_fnc=lambda x, y: ((x - y) ** 2)):
        self.N = N
        self.rho = rho
        self.eps = eps
        self.T = t_max
        self.solver = solver
        # self.dist_fnc = dist_fnc

    def __call__(self, online_net, target_net, score_net, images, labels, augment_pipe=None):
        # sample random integer n ~ U[1, N-1]
        n = randint(1, self.N-1)
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        t, t_prev = torch.tensor(self.t(n+1), device=images.device), torch.tensor(self.t(n), device=images.device)
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        noise = torch.randn_like(y) * t

        y_t = y + noise
        solver_dy = self.solver(score_net, y_t, t, t_prev, labels)
        y_t_prev = y_t + (t_prev - t) * solver_dy
        weight = 1.  # This could be some weighting function.

        # loss = weight * self.dist_fnc(online_net(y_t, t), target_net(y_t_prev, t))
        loss = weight * ((online_net(y_t, t) - target_net(y_t_prev, t)) ** 2)
        print(loss.mean().item())
        return loss

    def t(self, n):
        """Equation (6) from Consistency Models."""
        rho_inv = 1. / self.rho
        start = self.eps ** rho_inv
        end = self.T ** rho_inv
        scale = (n - 1) / (self.N - 1)
        t = (start + scale * (end - start)) ** self.rho
        return t

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
