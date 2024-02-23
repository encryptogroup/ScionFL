#!/usr/bin/env python3

from mpc.utils import *
from mpc.configuration import *


def mpc_mult(x, y, config: Configuration):
    is_preshared = isinstance(y, Sharing_PP)

    # offline phase

    # generate lambda sharing of inputs
    s_x = prepare_share_decimals_ext_PP(config.device, config.nshares, x.shape)
    s_x_lambda = sum(s_x.lambdas)

    if is_preshared:
        s_y = y
    else:
        s_y = prepare_share_decimals_ext_PP(config.device, config.nshares, y.shape)
    s_y_lambda = sum(s_y.lambdas)

    # compute shared inner product of x and y lambdas
    gamma_lambdas = s_x_lambda * s_y_lambda
    s_gamma = share_decimals_ext(config.device, gamma_lambdas, config.nshares, gamma_lambdas.shape)

    # generate random number for masking
    r = random_decimal_ext(config.device, gamma_lambdas.shape)
    s_r = share_decimals_ext(config.device, r, config.nshares, r.shape)
    r_truncated = truncate(r)

    # online phase

    # complete sharing based on actual inputs
    s_x = Sharing_PP(x - s_x_lambda, s_x.lambdas)
    if not is_preshared:
        s_y = Sharing_PP(y - s_y_lambda, s_y.lambdas)

    z_r = []
    t1 = s_x.m * s_y.m
    for j in range(config.nshares):
        t2 = s_x.m * s_y.lambdas[j]
        t3 = s_y.m * s_x.lambdas[j]

        t = t1 + t2 + t3

        z_r.append(t + s_gamma.shares[j] - s_r.shares[j])

        # t1 should only be considered once
        if j == 0:
            t1 = 0

    return truncate(sum(z_r)) + r_truncated
