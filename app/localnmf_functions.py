#@markdown #Step 1 NMF Demixing: Specify key parameters for the NMF demixing stage
import torch
import jax
import localnmf 
from localnmf import superpixel_analysis_ring
import scipy
import scipy.sparse
import os
import numpy as np
import sys
import datetime

def display(msg):
        """
        Printing utility that logs time and flushes.
        """
        tag = '[' + datetime.datetime.today().strftime('%y-%m-%d %H:%M:%S') + ']: '
        sys.stdout.write(tag + msg + '\n')
        sys.stdout.flush()


def run_localnmf_demixing(outdir, localnmf_params, a=None):
    '''
    Inputs: 
        - outdir: string. os path describing the location of the output directory (where results are saved)
        - localnmf_params: dict containing all the relevant user-tunable parameters for localnmf demixing
        - 'a' (optional): matrix of shape (d1, d2, K), where d1, d2 are the FOV dimensions and K is the number of neural signals
    
    '''
#This specifies the number of times we run the NMF algorithm on the data. If num_passes = 2 that means we run it once on the PMD data, then subtract the signals and 
#re-run on the residual
    num_passes =localnmf_params['num_passes'] #1 #@param {type:"slider", min:1, max:4, step:1}
    init=['lnmf' for i in range(num_passes)]
    

#This is the data structure we use to pass the data into the dictionary
    if a is not None:
        custom_init = dict()
        custom_init['a'] = a
        init[0] = 'custom'
    else:
        custom_init = None

    cut_off_point=[localnmf_params['superpixels_corr_thr'][i] for i in range(len(localnmf_params['superpixels_corr_thr']))]
    length_cut=localnmf_params['length_cut']
    th=localnmf_params['length_cut']

    corr_th_fix=localnmf_params['corr_th_fix'] 
    switch_point = localnmf_params['switch_point']
    corr_th_fix_sec = localnmf_params['corr_th_fix_sec']
    corr_th_del = localnmf_params['corr_th_del']

    max_allow_neuron_size= localnmf_params['max_allow_neuron_size'] #0.15
    merge_corr_thr= localnmf_params['merge_corr_thr']
    merge_overlap_thr= localnmf_params['merge_overlap_thr']
    r =  localnmf_params['r']




    ##Do not need to modify
    residual_cut = [0.5, 0.6, 0.6, 0.6]
    num_plane=1
    patch_size=[100,100]
    plot_en = False
    TF=False
    fudge_factor=1
    text=True
    max_iter=30
    init=init #lnmf specifies superpixel init
    max_iter_fin= 30 #Normally 30
    update_after= 10
    pseudo_1 = [0, 0, 0, 0]
    pseudo_2 = [1/20, 1/20,1/15, 0]
    skips=0
    update_type = "Constant" #Options here are 'Constant' or 'Full'
    custom_init = custom_init
    block_dims = None
    confidence = None
    spatial_thres = None
    frame_len = None
    model=None
    allowed_overlap = 40
    plot_mnmf = False
    sb = True
    pseudo_corr = [0, 0, 3/4, 3/4]
    plot_debug = False
    denoise = [False for i in range(max_iter)]
    for k in range(max_iter):
      if k > 0 and k % 20 == 0:
        denoise[k] = True
    batch_size = 100

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    print("the outdir in the nmf code is {}".format(outdir))
    input_file = os.path.join(outdir, "decomposition.npz")
    data = np.load(input_file, allow_pickle=True)
    U_sparse = scipy.sparse.csr_matrix(
            (data['U_data'], data['U_indices'], data['U_indptr']),
            shape=data['U_shape']
        ).tocoo()
    order = data.get('fov_order', np.array("C")).item()
    shape = data['U_shape']
    d1,d2 = data['fov_shape']
    R = data['R']
    s = data['s']
    Vt = data['Vt']
    T = Vt.shape[1]
    V = R.dot(s[:, None] *Vt)
    dims = (d1, d2, T)
    U_r = np.array(U_sparse.todense()).reshape((d1, d2,-1), order=order)

    # %load_ext line_profiler 
    try:
        torch.cuda.empty_cache()
        jax.clear_backends()
        rlt = superpixel_analysis_ring.demix_whole_data_robust_ring_lowrank(U_r,\
                                    V,r, cut_off_point,\
                                        length_cut, th, num_passes,\
                                        residual_cut, corr_th_fix,\
                                          corr_th_fix_sec, corr_th_del, switch_point,\
                                        max_allow_neuron_size, merge_corr_thr,\
                                        merge_overlap_thr, num_plane,\
                                        patch_size, plot_en, TF, \
                                        fudge_factor, text, max_iter,\
                                        max_iter_fin, update_after, \
                                        pseudo_1, pseudo_2, skips, update_type, init=init,\
                                        block_dims=block_dims, frame_len=frame_len,\
                                        confidence=confidence, spatial_thres=spatial_thres,\
                                                          model=model,custom_init=custom_init,\
                                                                    allowed_overlap=allowed_overlap, \
                                                                          plot_mnmf = plot_mnmf,\
                                                                          sb=sb, pseudo_corr = pseudo_corr, plot_debug = plot_debug,\
                                                                        denoise = denoise, device = device, batch_size = batch_size)


        display("Clearing memory from run")
        torch.cuda.empty_cache()
        jax.clear_backends()
    except Exception as e:
        print("\n \n \n")
        display("--------ERROR GENERATED, DETAILS BELOW-----")
        display("Unexpected error, please report")
        jax.clear_backends()
        torch.cuda.empty_cache()
        display("Cleared backends")
        print(e)
        display("Please re-run the pipeline starting from motion correction.")

