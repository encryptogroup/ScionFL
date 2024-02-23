#!/usr/bin/env python3

import pathlib

path = pathlib.Path(__file__).parent.resolve()

import sys

sys.path.insert(0, str(path) + "/../compressors")

import torch

from stochastic_quantization import StochasticQuantizationSender, StochasticQuantizationReceiver
from hadamard import HadamardSender, HadamardReceiver
from kashin import KashinSender, KashinReceiver

from mpc.simulation import simulate
from mpc.configuration import Configuration

##############################################################################
##############################################################################

SEED = 42

### supported algs
Algs = ['sq', 'hadamard', 'kashin']

### rotation-based algs
RAlgs = ['hadamard', 'kashin']

def benchmark_global(ntrials, nclients, dim, alg, nshares, configs):
    ncombinations = len(nshares) * len(configs)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    device = configs[0].device

    ### instantiate SQ
    sq_sender = StochasticQuantizationSender(device)
    sq_receiver = StochasticQuantizationReceiver(device)

    ### instantiate Hadamard
    hadamard_sender = HadamardSender(device)
    hadamard_receiver = HadamardReceiver(device)

    ### instantiate Kashin
    kashin_sender = KashinSender(device=device, eta=0.9, delta=1.0, pad_threshold=0.85, niters=3)
    kashin_receiver = KashinReceiver(device)

    ### Total NMSE
    NMSE = 0
    sim_NMSE = [0] * ncombinations

    ### distribution
    distribution = torch.distributions.LogNormal(0, 1)

    for trial in range(ntrials):

        ### all clients use the same seed for rotation-based algs
        if alg in RAlgs:
            trial_seed = SEED + trial

        ### original
        ovec = torch.zeros(dim).to(device)

        ### original sum of norms
        ovec_snorm = 0

        ### reconstructed
        rvec = torch.zeros(dim).to(device)

        ### draw client vectors from the given distribution
        client_vectors = distribution.sample([nclients, dim]).to(device)

        ### dimension of the bit-vector
        if alg == 'sq':
            d_bitvec = dim
        elif alg == 'hadamard':
            d_bitvec = hadamard_sender.dimension(dim)
        elif alg == 'kashin':
            d_bitvec = kashin_sender.dimension(dim)

        ### processed vecs (i.e., hadamard or kashin)
        if alg in RAlgs:
            client_pvecs = torch.zeros([nclients, d_bitvec]).to(device)

        ### accumulated quantized updates at the ps
        global_steps = torch.zeros(d_bitvec).to(device)

        ### preprocessing required
        if alg in RAlgs:

            ### init global scales
            smin, smax = 1e9, -1e9

            for client in range(nclients):

                if alg == 'hadamard':
                    client_pvecs[client] = hadamard_sender.randomized_hadamard_transform(client_vectors[client].clone(),
                                                                                         trial_seed)

                if alg == 'kashin':
                    client_pvecs[client] = kashin_sender.kashin_coefficients(client_vectors[client].clone(), trial_seed)

                smin = min(smin, client_pvecs[client].min())
                smax = max(smax, client_pvecs[client].max())

        ### preprocessing not required
        else:

            smin = client_vectors.min()
            smax = client_vectors.max()

        '''
        -- preliminary networking happens here.
        After all clients ready for quantization, they share their scales
        '''

        steps = torch.zeros([nclients, d_bitvec]).to(device)

        for client in range(nclients):

            ### for comparison - original vectors
            ovec += client_vectors[client]
            ovec_snorm += torch.norm(client_vectors[client], 2) ** 2

            ### quantization
            if alg in RAlgs:
                steps[client], step_size, start = sq_sender.compress(client_pvecs[client], 1, smin, smax)
            else:
                steps[client], step_size, start = sq_sender.compress(client_vectors[client], 1, smin, smax)

            ##################################################################
            ##################################################################
            '''
            -- This is where the networking between the clients and the PSs happens.
            -- ``steps'' should be converted to a bit vector and sent.
            -- ``step_size'' and ``start'' are floats and sent accuratly. 
            -- In our notations when nbits = 1: S_min = start and S_max = start+step_size 
            '''
            ##################################################################
            ##################################################################

            ''' accumulate votes - communication among PSs happens here '''
            global_steps += steps[client]

            ### original
        ovec /= nclients

        ### original sum of norms
        ovec_snorm /= nclients

        ### average steps
        global_steps /= nclients

        ### reconstruct vector before inverse processing
        reconstructed_pvec = sq_receiver.decompress(global_steps, step_size, start)

        sim_reconstructed_pvecs = []
        for config in configs:
            for nshare in nshares:
                config_nshare = Configuration(config.scales, config.computation, config.aggregation, config.mult_protocol, nshare, config.device)
                sim_reconstructed_pvecs.append(simulate(steps, start, step_size, config_nshare))

        ### ready for inverse transform
        sim_rvecs = []
        if alg in RAlgs:

            if alg == 'hadamard':
                rvec = hadamard_receiver.randomized_inverse_hadamard_transform(reconstructed_pvec, trial_seed)[:dim]
                for sim_reconstructed_pvec in sim_reconstructed_pvecs:
                    sim_rvecs.append(hadamard_receiver.randomized_inverse_hadamard_transform(sim_reconstructed_pvec, trial_seed)[
                           :dim])

            if alg == 'kashin':
                rvec = kashin_receiver.decompress(reconstructed_pvec, trial_seed)[:dim]
                for sim_reconstructed_pvec in sim_reconstructed_pvecs:
                    sim_rvecs.append(kashin_receiver.decompress(sim_reconstructed_pvec, trial_seed)[:dim])

        else:

            rvec = reconstructed_pvec
            for sim_reconstructed_pvec in sim_reconstructed_pvecs:
                sim_rvecs.append(sim_reconstructed_pvec)

        ### compute NMSE
        NMSE += torch.norm(ovec - rvec, 2) ** 2 / ovec_snorm
        for i in range(ncombinations):
            sim_NMSE[i] += torch.norm(ovec - sim_rvecs[i], 2) ** 2 / ovec_snorm

    print("Exact NMSE:\t{}".format(NMSE / ntrials), "\t", ntrials, "\t", nclients, "\t", dim, "\t", alg)
    combination = 0
    for config in configs:
        for nshare in nshares:
            print("Sim NMSE:\t{}".format(sim_NMSE[combination] / ntrials), "\t", repr(config), "\t", nshare, "\t", alg)
            combination+=1


def benchmark_local(ntrials, nclients, dim, alg, nshares, configs):
    ncombinations = len(nshares) * len(configs)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    ### device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    ### instantiate SQ
    sq_sender = StochasticQuantizationSender(device)
    sq_receiver = StochasticQuantizationReceiver(device)

    ### instantiate Hadamard
    hadamard_sender = HadamardSender(device)
    hadamard_receiver = HadamardReceiver(device)

    ### instantiate Kashin
    kashin_sender = KashinSender(device=device, eta=0.9, delta=1.0, pad_threshold=0.85, niters=3)
    kashin_receiver = KashinReceiver(device)

    ### Total NMSE
    NMSE = 0
    sim_NMSE = [0] * ncombinations

    ### distribution
    distribution = torch.distributions.LogNormal(0, 1)

    ### debug for potential overflows: run more trials than necessary, discard failures
    ntrials_buffer = int(ntrials + 0.3 * ntrials)
    valid_trials = 0

    for trial in range(ntrials_buffer):

        ### all clients use the same seed for rotation-based algs
        if alg in RAlgs:
            trial_seed = SEED + trial

        ### original
        ovec = torch.zeros(dim).to(device)

        ### original sum of norms
        ovec_snorm = 0

        ### reconstructed
        rvec = torch.zeros(dim).to(device)

        ### draw client vectors from the given distribution
        client_vectors = distribution.sample([nclients, dim]).to(device)

        ### dimension of the bit-vector
        if alg == 'sq':
            d_bitvec = dim
        elif alg == 'hadamard':
            d_bitvec = hadamard_sender.dimension(dim)
        elif alg == 'kashin':
            d_bitvec = kashin_sender.dimension(dim)

        ### processed vecs (i.e., hadamard or kashin)
        if alg in RAlgs:
            client_pvecs = torch.zeros([nclients, d_bitvec]).to(device)

        ### accumulated quantized updates at the ps
        reconstructed_pvec = torch.zeros(d_bitvec).to(device)

        ### preprocessing required
        if alg in RAlgs:

            for client in range(nclients):

                if alg == 'hadamard':
                    client_pvecs[client] = hadamard_sender.randomized_hadamard_transform(client_vectors[client].clone(),
                                                                                         trial_seed)

                if alg == 'kashin':
                    client_pvecs[client] = kashin_sender.kashin_coefficients(client_vectors[client].clone(), trial_seed)

        steps = torch.zeros([nclients, d_bitvec]).to(device)
        step_size = torch.zeros([nclients, 1]).to(device)
        start = torch.zeros([nclients, 1]).to(device)

        for client in range(nclients):

            ### for comparison - original vectors
            ovec += client_vectors[client]
            ovec_snorm += torch.norm(client_vectors[client], 2) ** 2

            ### quantization
            if alg in RAlgs:
                steps[client], step_size[client], start[client] = sq_sender.compress(client_pvecs[client], 1)
            else:
                steps[client], step_size[client], start[client] = sq_sender.compress(client_vectors[client], 1)

            ##################################################################
            ##################################################################
            '''
            -- This is where the networking between the clients and the PSs happens.
            -- ``steps'' should be converted to a bit vector and sent.
            -- ``step_size'' and ``start'' are floats and sent accuratly. 
            -- In our notations when nbits = 1: S_min = start and S_max = start+step_size 
            '''
            ##################################################################
            ##################################################################

            ''' accumulate votes - communication among PSs happens here '''
            reconstructed_pvec += sq_receiver.decompress(steps[client], step_size[client], start[client])

        ### original
        ovec /= nclients

        ### original sum of norms
        ovec_snorm /= nclients

        ### average reconstructed
        reconstructed_pvec /= nclients

        sim_reconstructed_pvecs = []
        for config in configs:
            for nshare in nshares:
                config_nshare = Configuration(config.scales, config.computation, config.aggregation, config.mult_protocol, nshare, config.device)
                sim_reconstructed_pvecs.append(simulate(steps, start, step_size, config_nshare))

        ### ready for inverse transform
        sim_rvecs = []
        if alg in RAlgs:

            if alg == 'hadamard':
                rvec = hadamard_receiver.randomized_inverse_hadamard_transform(reconstructed_pvec, trial_seed)[:dim]
                for sim_reconstructed_pvec in sim_reconstructed_pvecs:
                    sim_rvecs.append(hadamard_receiver.randomized_inverse_hadamard_transform(sim_reconstructed_pvec, trial_seed)[
                           :dim])

            if alg == 'kashin':
                rvec = kashin_receiver.decompress(reconstructed_pvec, trial_seed)[:dim]
                for sim_reconstructed_pvec in sim_reconstructed_pvecs:
                    sim_rvecs.append(kashin_receiver.decompress(sim_reconstructed_pvec, trial_seed)[:dim])

        else:

            rvec = reconstructed_pvec
            for sim_reconstructed_pvec in sim_reconstructed_pvecs:
                sim_rvecs.append(sim_reconstructed_pvec)

        ### compute NMSE
        result_NMSE = torch.norm(ovec - rvec, 2) ** 2 / ovec_snorm
        result_sim_NMSE = [0] * ncombinations
        abort = False
        for i in range(ncombinations):
            result = torch.norm(ovec - sim_rvecs[i], 2) ** 2 / ovec_snorm
            result_sim_NMSE[i] += result

            # abort if there is an overflow (should not happen)
            if result > 10000:
                abort = True

        if abort:
            continue
        else:
            NMSE += result_NMSE
            for i in range(ncombinations):
                sim_NMSE[i] += result_sim_NMSE[i]

            valid_trials += 1
            if valid_trials == ntrials:
                break

    print("Exact NMSE:\t{}".format(NMSE / ntrials), "\t", ntrials, "\t", nclients, "\t", dim, "\t", alg)
    combination = 0
    for config in configs:
        for nshare in nshares:
            print("Sim NMSE:\t{}".format(sim_NMSE[combination] / ntrials), "\t", repr(config), "\t", nshare, "\t", alg)
            combination += 1
