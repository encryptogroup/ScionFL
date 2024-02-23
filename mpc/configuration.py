#!/usr/bin/env python3

from enum import Enum


class Scales(Enum):
    Local = 1
    Global = 2


class Computation(Enum):
    Exact = 1
    Approximate = 2


class Aggregation(Enum):
    Exact = 1
    SepAgg = 2


class MultProtocol(Enum):
    Float = 1
    SFN = 2
    MPC = 3


class Configuration:
    def __init__(self, scales: Scales, computation: Computation, aggregation: Aggregation, mult_protocol: MultProtocol, nshares, device):
        self.scales = scales
        self.computation = computation
        self.aggregation = aggregation
        self.mult_protocol = mult_protocol
        self.nshares = nshares
        self.device = device

    def __repr__(self):
        return self.scales.name + "\t" + self.computation.name + "\t" + self.aggregation.name + "\t" + self.mult_protocol.name + "\t" + str(self.nshares)
