import torch
import numpy as np
import os

def FDD_Extension(u, dt):
    du = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dt)
    ddu = (u[:, 2:, :] - 2 * u[:, 1:-1, :] + u[:, :-2, :]) / (dt ** 2)
    return du, ddu
