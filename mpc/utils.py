#!/usr/bin/env python3

from collections import namedtuple
import functools
import operator

from mpc.constants import *

Sharing = namedtuple("Sharing", "shares")
Sharing_PP = namedtuple("Sharing_PP", "m lambdas")


def random_decimal_ext(device, size=[]):
    return torch.randint(EXT_RING_MIN, EXT_RING_MAX, size, dtype=EXT_SFN_TYPE).to(device)


def random_decimal(device, size=[]):
    return torch.randint(RING_MIN, RING_MAX, size, dtype=SFN_TYPE).to(device)


def random_bit(device, size=[]):
    return torch.randint(0, BIN_MOD, size, dtype=BOOL_TYPE).to(device)


def zeros(device, size=[]):
    return torch.zeros(size, dtype=SFN_TYPE).to(device)


def prepare_share_decimals_ext_PP(device, nshares, size=[]):
    lambdas = []
    for i in range(nshares):
        lambdas.append(random_decimal_ext(device, size))

    return Sharing_PP(zeros(device, size), lambdas)


def share_decimals_ext_PP(device, decimals, nshares, size=[]):
    lambdas = []
    for i in range(nshares):
        lambdas.append(random_decimal_ext(device, size))

    return Sharing_PP(decimals - sum(lambdas), lambdas)


def share_decimals_ext(device, decimals, nshares, size=[]):
    shares = []
    for i in range(nshares - 1):
        shares.append(random_decimal_ext(device, size))

    shares.append(decimals - sum(shares))

    return Sharing(shares)


def share_bits_PP(device, bits, nshares, size):
    lambdas = []
    for i in range(nshares):
        lambdas.append(random_bit(device, size))

    return Sharing_PP(to_bool((bits + sum(lambdas)) % BIN_MOD), lambdas)


def share_bits(device, bits, nshares, size):
    shares = []
    for i in range(nshares - 1):
        shares.append(random_bit(device, size))

    shares.append(to_bool((bits + sum(shares)) % BIN_MOD))

    return Sharing(shares)


def sum_shares_PP(s_share: Sharing_PP):
    lambdas = []
    for i in range(len(s_share.lambdas)):
        lambdas.append(sum(s_share.lambdas[i]))

    return Sharing_PP(sum(s_share.m), lambdas)


def mulreduce(*args):
    return functools.reduce(operator.mul, args)


def local_mul(x, y):
    x = float_to_sfn64(x)

    return truncate(x * y)


def truncate(input):
    return (input >> SFN_FRACTION).type(EXT_SFN_TYPE)


def to_bool(input):
    return input.type(BOOL_TYPE)


def to_int(input):
    return input.type(SFN_TYPE)


def to_int64(input):
    return input.type(EXT_SFN_TYPE)


def to_SFN(input):
    return input * SFN_FACTOR


def float_to_sfn(input):
    return to_int(to_SFN(input))


def float_to_sfn64(input):
    return to_int64(to_SFN(input))


def sfn_to_float(input):
    return input / SFN_FACTOR
