#!/usr/bin/env python3

"""
Based on: 
(1) https://arxiv.org/pdf/math/0611343.pdf
Lyubarskii, Yurii, and Roman Vershynin. "Uncertainty principles and vector quantization." 
IEEE Transactions on Information Theory 56.7 (2010): 3491-3501.

(2) https://arxiv.org/pdf/2002.08958.pdf
Safaryan, Mher, Egor Shulgin, and Peter Richtárik. "Uncertainty Principle for Communication Compression in Distributed and Federated Learning and the Search for an Optimal Compressor." 
arXiv preprint arXiv:2002.08958 (2020).
"""

import pathlib
path = pathlib.Path(__file__).parent.resolve()

import sys
sys.path.insert(0, str(path) + "/../")

import torch
import numpy as np

from hadamard import HadamardSender, HadamardReceiver

##############################################################################
##############################################################################

class KashinSender(HadamardSender, HadamardReceiver):

    
    def __init__(self, device, eta, delta, pad_threshold, niters):

        HadamardSender.__init__(self, device=device)
        HadamardReceiver.__init__(self, device=device)
        
        self.eta = eta
        self.delta=delta
        self.pad_threshold = pad_threshold
        self.niters = niters
        
    
    def dimension(self, dim):
        
        padded_dim = dim
        
        if not dim & (dim-1) == 0:
            padded_dim = int(2**(np.ceil(np.log2(dim))))
            if dim / padded_dim > self.pad_threshold:
                padded_dim = 2*padded_dim
        else:
            padded_dim = 2*dim
            
        return padded_dim

    
    def kashin_coefficients(self, vec, seed):
        
        dim = vec.numel()
        padded_dim = self.dimension(dim)
               
        kashin_coefficients_vec = torch.zeros(padded_dim, device=self.device)
        padded_x = torch.zeros(padded_dim, device=self.device)
        
        M = torch.norm(vec) / np.sqrt(self.delta * padded_dim)
        
        for i in range(self.niters):
    
            padded_x[:] = 0
            padded_x[:dim] = vec  
            padded_x = self.randomized_hadamard_transform(padded_x, seed)
            
            b = padded_x   
            b_hat = torch.clamp(b, min=-M, max=M)
                    
            kashin_coefficients_vec = kashin_coefficients_vec + b_hat
            
            if i < self.niters - 1:
            
                b_hat = self.randomized_inverse_hadamard_transform(b_hat, seed)
                vec = vec - b_hat[:dim]
                
                M = self.eta * M

        return kashin_coefficients_vec              
    
##############################################################################
##############################################################################
       
class KashinReceiver(HadamardReceiver):

    
    def __init__(self, device='cpu'):

        HadamardReceiver.__init__(self, device=device)
                            
    def decompress(self, vec, seed):
                
        return self.randomized_inverse_hadamard_transform(vec, seed)
   
##############################################################################
############################################################################## 
