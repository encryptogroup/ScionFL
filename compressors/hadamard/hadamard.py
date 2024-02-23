#!/usr/bin/env python3

"""
Based on: https://proceedings.mlr.press/v70/suresh17a/suresh17a.pdf
Suresh, Ananda Theertha, et al. "Distributed mean estimation with limited communication." International conference on machine learning. PMLR, 2017.
"""



import pathlib
path = pathlib.Path(__file__).parent.resolve()

import sys
sys.path.insert(0, str(path) + "/../")

import torch
import numpy as np

##############################################################################
##############################################################################

class Hadamard:


    def __init__(self, device='cpu'):

        self.device = device   
        self.prng = torch.Generator(device=device)
        

    def hadamard(self, vec):
        
        d = vec.numel()
        if d & (d-1) != 0:
            raise Exception("input numel must be a power of 2")
          
        h = 2
        while h <= d:        
            hf = h//2
            vec = vec.view(d//h,h)
            vec[:,:hf]  = vec[:,:hf] + vec[:,hf:2*hf]
            vec[:,hf:2*hf] = vec[:,:hf] - 2*vec[:,hf:2*hf]
            h *= 2   
        vec /= np.sqrt(d)
        
        return vec.view(-1)


    def random_diagonal(self, size, seed):
        
        self.prng.manual_seed(seed)
        return 2 * torch.bernoulli(torch.ones(size=(size,), device=self.device) / 2, generator=self.prng) - 1
    
##############################################################################
##############################################################################
        
class HadamardSender(Hadamard):

    
    def __init__(self, device='cpu'):

        Hadamard.__init__(self, device=device)
                
            
    def randomized_hadamard_transform(self, vec, seed):

        dim = vec.numel()
        
        if not dim & (dim - 1) == 0:
            
            padded_dim = int(2**(np.ceil(np.log2(dim))))
            padded_vec = torch.zeros(padded_dim, device=self.device)
            padded_vec[:dim] = vec
            
            padded_vec = padded_vec * self.random_diagonal(padded_vec.numel(), seed)
            padded_vec = self.hadamard(padded_vec)
            
            return padded_vec
        
        else:   
            
            vec = vec * self.random_diagonal(vec.numel(), seed)
            vec = self.hadamard(vec)
            
            return vec

    def dimension(self, dim):
        
        if not dim & (dim - 1) == 0:          
            return int(2**(np.ceil(np.log2(dim))))
        else:   
            return dim
        
##############################################################################
##############################################################################
       
class HadamardReceiver(Hadamard):

    
    def __init__(self, device='cpu'):

        Hadamard.__init__(self, device=device)
                
            
    def randomized_inverse_hadamard_transform(self, vec, seed):
        
        vec = self.hadamard(vec)
        vec = vec * self.random_diagonal(vec.numel(), seed)
        
        return vec
   
##############################################################################
############################################################################## 

