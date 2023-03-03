#@markdown #Step 1 NMF Demixing: Specify key parameters for the NMF demixing stage
import torch
import torch_sparse
import jax
import localnmf 
from localnmf import superpixel_analysis_ring
import scipy
import scipy.sparse
import os
import numpy as np
import sys
import datetime
import math



def get_single_pixel_corr_img(U, R, V, row_index, batch_size=100):
    '''
    Returns the correlation between a single pixel of the URV PMD decomposition and all other pixels
    Key assumption: V has orthonormal rows
    Inputs: 
        U: torch_sparse.tensor, shape (d1*d2, R) 
        R: torch.Tensor (R, R) shaped matrix
        V: torch.Tensor (R, T) shaped matrix
        row_index: int. the pixel (the row of U) which we are interested in analyzing
    '''
    device = V.device
    V_mean = torch.mean(V, dim = 1, keepdim=True)
    RV_mean = torch.matmul(R, V_mean) 
    URV_mean = torch_sparse.matmul(U, RV_mean) 
    s = torch.matmul(torch.ones((1,V.shape[1]), device=device), V.t())
    
    ind_torch = torch.arange(row_index, row_index+1, step=1, device=device)
    single_pixel = torch_sparse.index_select(U, 0, ind_torch)
    centered_pixel_trace = torch_sparse.matmul(single_pixel, R) - torch.matmul(URV_mean[[row_index], :], s)
    num_iters = math.ceil(R.shape[1] / batch_size)
    
    normalizers = torch.zeros((U.sparse_sizes()[0], 1), device=device)
    cumulator = torch.zeros((U.sparse_sizes()[0], 1), device=device)
    for k in range(num_iters):
        start = batch_size*k
        end = min(R.shape[1], start + batch_size)
        
        R_cropped = R[:, start:end]
        s_cropped = s[:, start:end]
        centered_pixel_trace_cropped = centered_pixel_trace[:, start:end]
        UR_cropped = torch_sparse.matmul(U, R_cropped)
        URV_mean_Vbasis_cropped = torch.matmul(URV_mean, s_cropped)
        centered_UR_cropped = UR_cropped - URV_mean_Vbasis_cropped
        
        all_normalizers = torch.sum(centered_UR_cropped * centered_UR_cropped, dim = 1, keepdim=True)
        normalizers += all_normalizers
        
        row_wise_dots = torch.sum(centered_pixel_trace_cropped * centered_UR_cropped, dim = 1, keepdim=True)
        cumulator += row_wise_dots
        
        
    normalizers = torch.sqrt(normalizers)
    normalizers = normalizers * normalizers[[row_index],0]
    normalizers[normalizers == 0] = 1
    return cumulator / normalizers
    
    
    
    
    

def display(msg):
        """
        Printing utility that logs time and flushes.
        """
        tag = '[' + datetime.datetime.today().strftime('%y-%m-%d %H:%M:%S') + ']: '
        sys.stdout.write(tag + msg + '\n')
        sys.stdout.flush()