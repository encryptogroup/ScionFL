#!/usr/bin/env python3

import pathlib
path = pathlib.Path(__file__).parent.resolve()

import sys
sys.path.insert(0, str(path) + "/../")

import torch

##############################################################################
##############################################################################

class StochasticQuantizationSender:


    def __init__(self, device='cpu'):

        self.device = device


    def compress(self, vec, nbits, vec_min=None, vec_max=None):
        
        ### global or local?
        if vec_max is None:
            vec_max = vec.max()
        if vec_min is None:
            vec_min = vec.min()
        
        ### step size
        step_size = (vec_max-vec_min)/(2**nbits-1) 
        
        ### number of steps from the minimum
        steps = (vec-vec_min)/step_size 
        
        ### floor for stochastic quantization
        steps_floor = torch.floor(steps)
        
        ### stochastic quantization
        steps = steps_floor + torch.bernoulli(steps-steps_floor)

        return steps, step_size, vec_min

##############################################################################
##############################################################################

class StochasticQuantizationReceiver:


    def __init__(self, device='cpu'):

        self.device = device


    def decompress(self, steps, step_size, start):

        return start+steps*step_size

##############################################################################
##############################################################################

