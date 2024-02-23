#!/usr/bin/env python3

import itertools

from mpc.utils import *
from mpc.configuration import *
from mpc.protocols.mult import mpc_mult


def sim_mul(in1, in2, config):
    if config.mult_protocol == MultProtocol.Float:
        result = in1 * in2
    elif config.mult_protocol == MultProtocol.SFN:
        # to accommodate approximate computations (which include .5 values for uneven nshares), use scaled input
        result = truncate(float_to_sfn64(in1) * in2)
    elif config.mult_protocol == MultProtocol.MPC:
        result = mpc_mult(float_to_sfn(in1), in2, config)

    return result


def sim_div(in1, in2, config):
    if config.mult_protocol == MultProtocol.Float:
        return in1 / in2
    else:
        return to_int(in1 / in2)


def prepare_result(result, config):
    if config.mult_protocol == MultProtocol.Float:
        return result
    else:
        return sfn_to_float(to_int(result))


def simulate(bits, start, step_size, config: Configuration):
    nclients = bits.shape[0]
    nbits = bits.shape[1]

    bits_int = to_bool(bits)

    # floats must be in SFN format for SFN / MPC simulation
    if not config.mult_protocol == MultProtocol.Float:
        start = float_to_sfn(start)
        step_size = float_to_sfn(step_size)

    if not config.mult_protocol == MultProtocol.MPC:
        # to have comparable accuracy results regarding middle term approximation for DHM protocols, bump nshares
        nshares = config.nshares + 1

        s_bits = share_bits(config.device, bits_int, nshares, [nclients, nbits])

        t_sum = sum(s_bits.shares)

        if config.computation == Computation.Exact:
            t_middle = zeros(config.device, [nclients, nbits])
            for k in range(2, nshares):
                t_middle += (-2) ** (k - 1) * sum(
                    list(itertools.starmap(mulreduce, itertools.combinations(s_bits.shares, k))))

        elif config.computation == Computation.Approximate:
            t_middle = (nshares - 1) % BIN_MOD - nshares / 2

        t_product = ((-2) ** (nshares - 1)) * functools.reduce(operator.mul, s_bits.shares)

        # compute arithmetic value of bits from terms
        b = t_sum + t_middle + t_product

        if config.aggregation == Aggregation.Exact:
            if config.scales == Scales.Global:
                result = start + sim_div(sim_mul(sum(b), step_size, config), nclients, config)
            elif config.scales == Scales.Local:
                result = sim_div(sum(start) + sum(sim_mul(b, step_size, config)), nclients, config)

        elif config.aggregation == Aggregation.SepAgg:
            if config.scales == Scales.Local:
                result = sim_div(sum(start), nclients, config) + \
                         sim_div(sim_mul(sum(b), sum(step_size), config), nclients * nclients, config)
            else:
                raise Exception("SepAgg only works for local!")

    elif config.mult_protocol == MultProtocol.MPC:
        # share inputs in pre-processing model (b = m + lambda_1 + lambda_2 ... lambda_nshares)
        s_bits = share_bits_PP(config.device, bits_int, config.nshares, [nclients, nbits])

        if config.scales == Scales.Global:
            s_step_size = share_decimals_ext_PP(config.device, step_size, config.nshares)
        elif config.scales == Scales.Local:
            s_step_size = share_decimals_ext_PP(config.device, step_size, config.nshares, [nclients, 1])

        # convert lambda parts of bit shares to arithmetic

        t_sum = sum(s_bits.lambdas)

        if config.computation == Computation.Exact:
            t_middle = zeros(config.device, [nclients, nbits])
            for k in range(2, config.nshares):
                t_middle += (-2) ** (k - 1) * sum(
                    list(itertools.starmap(mulreduce, itertools.combinations(s_bits.lambdas, k))))

        elif config.computation == Computation.Approximate:
            t_middle = config.nshares % BIN_MOD - (config.nshares + 1) / 2

        t_product = ((-2) ** (config.nshares - 1)) * functools.reduce(operator.mul, s_bits.lambdas)

        # arithmetic representation of lambda parts of bit shares
        Lambda_b = t_sum + t_middle + t_product

        # compute frequently used term
        om2mb = 1 - 2 * s_bits.m

        if config.aggregation == Aggregation.Exact:
            if config.scales == Scales.Global:
                # summed arithmetic representation of bit shares
                b = sum(s_bits.m + om2mb * Lambda_b)

                result = start + sim_div(sim_mul(b, s_step_size, config), nclients, config)

            elif config.scales == Scales.Local:
                Lambda_s = sum(s_step_size.lambdas)

                t1 = sum(s_bits.m * s_step_size.m)
                t2 = sum(om2mb * local_mul(Lambda_b, s_step_size.m))
                t3 = sum(s_bits.m * Lambda_s)
                t4 = sum(om2mb * sim_mul(Lambda_b, Lambda_s, config))

                result = sim_div(sum(start) + t1 + t3 + t2 + t4, nclients, config)

        elif config.aggregation == Aggregation.SepAgg:
            if config.scales == Scales.Local:
                # summed arithmetic representation of bit shares
                b = sum(s_bits.m + om2mb * Lambda_b)

                result = sim_div(sum(start), nclients, config) + \
                         sim_div(sim_mul(b, sum_shares_PP(s_step_size), config), nclients * nclients, config)

            else:
                raise Exception("SepAgg only works for local!")

    return prepare_result(result, config)
