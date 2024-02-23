#!/usr/bin/env python3

import torch
from mpc.configuration import *
from tests.NMSE_benchmark import benchmark_global, benchmark_local

##############################################################################
##############################################################################

if __name__ == '__main__':
    ntrials = 10
    nclients = [1, 10, 100, 1000, 10000]
    dims = [2**10, 2**15, 2**20]
    algs = ['sq', 'hadamard', 'kashin']
    nshares = [3, 4, 5, 6]
    computations = [Computation.Exact, Computation.Approximate]
    aggregations = [Aggregation.Exact, Aggregation.SepAgg]
    mult_protocols = [MultProtocol.MPC]

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    for dim in dims:
        for alg in algs:
            for nclient in nclients:
                configs_global = []
                configs_local = []
                for mult_protocol in mult_protocols:
                    for computation in computations:
                        for aggregation in aggregations:
                            if not aggregation == Aggregation.SepAgg:
                                configs_global.append(
                                    Configuration(Scales.Global, computation, aggregation, mult_protocol, 0, device))
                            configs_local.append(
                                Configuration(Scales.Local, computation, aggregation, mult_protocol, 0, device))
                benchmark_global(ntrials, nclient, dim, alg, nshares, configs_global)
                benchmark_local(ntrials, nclient, dim, alg, nshares, configs_local)
