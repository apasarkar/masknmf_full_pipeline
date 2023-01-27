import torch
import torch_sparse
import sys

import copy
#Misc imports
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import colorsys
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import shutil
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import skimage
from skimage import measure
from skimage import filters

import random
import numpy as np
import os
import skimage.io as io

import torch_sparse

import scipy
import scipy.sparse

#Ring-LocalNMF specific imports
from localnmf import superpixel_analysis_ring


import boto3
from botocore.config import Config
from botocore import UNSIGNED

from masknmf.engine.segmentation import segment_local_UV, filter_components_UV
from masknmf.detection.maskrcnn_detector import maskrcnn_detector





def _run_masknmf(data_folder, input_file, confidence, allowed_overlap, cpu_only,\
                block_dims_x, block_dims_y, frame_len, spatial_thresholds):

    input_file = os.path.join(data_folder, "decomposition.npz")

    data = np.load(input_file, allow_pickle=True)

    U_sparse = scipy.sparse.csr_matrix(
          (data['U_data'], data['U_indices'], data['U_indptr']),
          shape=data['U_shape']
      ).tocoo()

    order = data.get('fov_order', np.array("C")).item()
    U = np.array(U_sparse.todense())
    shape = data['U_shape']
    d1,d2 = data['fov_shape']
    R = data['R']
    s = data['s']
    Vt = data['Vt']
    T = Vt.shape[1]
    V_full = R.dot(s[:, None] *Vt)
    V = V_full
    dims = (d1, d2, T)
    U_r = U.reshape((d1, d2,-1), order=order)


    #This is where we create the neural network
    dir_path = "neuralnet_info" 

    #Specify where to save these outputs
    MASK_NMF_CONFIG = os.path.join(dir_path, "config_nn.yaml")
    MASK_NMF_MODEL = os.path.join(dir_path, "model_final.pth")

    #Specify where to retrieve the neural net data. In this case, in the apasarkar-public bucket on AWS S3
    bucket_loc = "apasarkar-public"
    config_file_name = "config.yaml"
    weights_file_name = "model_final.pth"



    if not os.path.isdir("neuralnet_info"):
      os.mkdir("neuralnet_info")


    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    s3.download_file(bucket_loc,
                  config_file_name,
                  MASK_NMF_CONFIG)
    s3.download_file(bucket_loc,
                  weights_file_name,
                  MASK_NMF_MODEL)



    model = maskrcnn_detector(MASK_NMF_MODEL,
                                MASK_NMF_CONFIG,
                                confidence,
                                allowed_overlap,
                                cpu_only, order=order)

    temporal_components = data.get('deconvolved_temporal')
    mixing_weights = data['R']

    ##END OF PARAMETER DEFINITION



    if temporal_components is None:
      raise ValueError("Deconvolution was not run on this PMD output. Re-run PMD with deconv")


    # Run Detection On Select Frames
    print("Performing MaskRCNN detection...")
    bin_masks, footprints, properties, _ = segment_local_UV(
      U_sparse,
      mixing_weights,
      temporal_components,
      tuple((d1, d2, temporal_components.shape[-1])),
      model,
      frame_len,
      block_size=(block_dims_x, block_dims_y),
      order=order
    )

    print("Filtering detected components...")
    keep_masks = filter_components_UV(
          footprints, bin_masks, properties,
          spatial_thresholds[0], spatial_thresholds[1])
    print(f"Filtering completed {keep_masks.shape} of {np.count_nonzero(keep_masks)} components retained")

    a_dense = np.asarray(footprints[:, keep_masks].todense())
    a = a_dense.reshape((d1, d2, -1), order=order)

    return a


def run_masknmf(data_folder, input_file, confidence, allowed_overlap, cpu_only,\
                block_dims_x, block_dims_y, frame_len, spatial_thresholds):
    try: 
        import torch
        import jax
        torch.cuda.empty_cache()
        jax.clear_backends()


        a = _run_masknmf(data_folder, input_file, confidence, allowed_overlap, cpu_only,\
                    block_dims_x, block_dims_y, frame_len, spatial_thresholds)



        torch.cuda.empty_cache()
        jax.clear_backends()

        return a
    except:
        print("\n \n \n")
        display("--------ERROR GENERATED, DETAILS BELOW-----")
        display("Unexpected error, please report")
        import jax
        import torch
        jax.clear_backends()
        torch.cuda.empty_cache()
        display("Cleared backends")
        print(e)
        display("Please re-run the pipeline starting from motion correction.")


    
