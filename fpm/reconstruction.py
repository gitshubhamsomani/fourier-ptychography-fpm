"""
Copyright (c) 2026, John Meshreki
All rights reserved.

john.meshreki@gmail.com

-------------------------------------
FPM EPRY reconstruction algorithm.
This module implements the core FPM reconstruction algorithm: embedded pupil function recovery [Ou et. al. 2014]. 
The main function is `AlterMin`, which performs the alternating minimization reconstruction. 
It relies on helper functions for the gradient descent update and Fourier projection.
"""
import torch
import matplotlib.pyplot as plt
import time
from fpm.utils import Ft, F
from fpm.logger import FPMLoggerObject

def GDUpdate_Multiplication_rank1(O, P, dpsi, Omax, cen, Ps, alpha, beta, step_size, optimize_P):
    Np = torch.tensor(P.shape)
    n1 = [int(cen[0] - torch.floor(Np[0] / 2)), int(cen[1] - torch.floor(Np[1] / 2))]
    n2 = [n1[0] + Np[0], n1[1] + Np[1]]

    O1 = O[n1[0]:n2[0], n1[1]:n2[1]]

    denom_P = torch.abs(P) ** 2 + alpha
    O_update = step_size * (1 / torch.max(torch.abs(P))) * torch.abs(P) * torch.conj(P) * dpsi / denom_P
    O[n1[0]:n2[0], n1[1]:n2[1]] = O1 + O_update

    if optimize_P:
        denom_O1 = torch.abs(O1) ** 2 + beta
        P_update = (1 / Omax) * torch.abs(O1) * torch.conj(O1) * dpsi / denom_O1 * Ps
        P = P + P_update

    return O, P


def Proj_Fourier_v2(psi0, I, I0, c):
    eps = torch.finfo(torch.float32).eps

    if psi0.ndim == 2:
        psi = F(torch.sqrt(I / c) * torch.exp(1j * torch.angle(psi0)))
    else:
        r = psi0.shape[2]
        psi = torch.zeros_like(psi0, dtype=torch.complex64)
        for m in range(r):
            psi[:, :, m] = F(torch.sqrt(I / c[m]) * psi0[:, :, m] / (torch.sqrt(I0) + eps))
    return psi


def AlterMin(I, No, Ns, P0, O0, use_default_P0, pupil_scale_fac, optimize_P, mode, Nled,
             tol, maxIter, minIter, monotone, StepSize, OP_alpha, OP_beta, display, Ps, device):
    
    logger = FPMLoggerObject.log

    I = I.to(device)
    Ns = Ns.to(device)
    O = O0.to(device)
    P = P0.to(device)
    Ps = P0.abs().gt(0).float().to(device)

    Nmy, Nmx, Nimg = I.shape
    Np = [Nmy, Nmx]

    cen0 = torch.round((torch.tensor(No, device=device).float() + 1) / 2).long()

    H0 = torch.ones(Np, dtype=torch.complex64, device=device)

    err1 = torch.tensor(float('inf'), device=device)
    err2 = torch.tensor(0.0, device=device)
    err = []

    scale = torch.ones((1, Nimg), device=device)

    logger.info("Starting AlterMin reconstruction")
    logger.info(
        "AlterMin settings: maxIter=%s, minIter=%s, tol=%s, monotone=%s, "
        "stepSize=%s, optimize_P=%s, mode=%s",
        maxIter,
        minIter,
        tol,
        monotone,
        StepSize,
        optimize_P,
        mode,
    )
    logger.info(
        "AlterMin tensor shapes: I=%s, Ns=%s, O0=%s, P0=%s",
        tuple(I.shape),
        tuple(Ns.shape),
        tuple(O.shape),
        tuple(P.shape),
    )
    logger.info("| iter |    rmse    |")
    logger.info("--------------------")

    for iter_idx in range(maxIter):
        err1 = err2.clone()
        err2 = torch.tensor(0.0, device=device)

        for m in range(Nimg):
            cen = cen0 - Ns[0, m, :]

            n1_y = int(cen[0] - Np[0] // 2)
            n1_x = int(cen[1] - Np[1] // 2)
            n2_y = n1_y + Np[0]
            n2_x = n1_x + Np[1]

            O_crop = O[n1_y:n2_y, n1_x:n2_x]

            Psi0 = O_crop * P * H0
            psi0 = Ft(Psi0)

            I_mea = I[:, :, m]
            I_est = torch.abs(psi0) ** 2

            Psi = Proj_Fourier_v2(psi0, I_mea, I_est, 1.0)

            dpsi = Psi - Psi0

            Omax = torch.abs(O[cen0[0], cen0[1]])

            O, P = GDUpdate_Multiplication_rank1(
                O, P, dpsi, Omax, cen, Ps,
                OP_alpha, OP_beta, StepSize, optimize_P
            )

            err2 = err2 + torch.sqrt(torch.sum((I_mea - I_est) ** 2))

        err_value = err2.item()
        err.append(err_value)
        logger.info("| %2d | %.2e |", iter_idx + 1, err_value)


        if monotone and iter_idx > minIter:
            if err2 > err1:
                logger.info(
                    "Stopping early because monotonicity was violated at iteration %d: "
                    "current error %.6e > previous error %.6e",
                    iter_idx + 1,
                    err2.item(),
                    err1.item(),
                )
                break

    if mode == 'fourier':
        O = Ft(O)

    logger.info("Finished AlterMin reconstruction after %d iteration(s)", len(err))

    return O, P, None, err, None, Ns