.. maskNMF summary

Summary
=======
This is a `Dash app <https://github.com/apasarkar/masknmf_full_pipeline>`_ for GPU-accelerated interactive data analysis of functional neuro-imaging data using the maskNMF pipeline. Through the app you can motion correct (non-rigid), compress (PMD), and demix (NMF variant) imaging data. We provide convenient visualization options at each of these stages along the way. You can export the compressed and demixed results easily in a format for local viewing (Napari) and further analysis.

The app is a tool that links many of the main steps for processing functional imaging data with a convenient interface for fast processing and visual inspection:

- `*Accelerated NoRMCoRRE* <https://github.com/apasarkar/jnormcorre>`_
- `*Accelerated SVD-like data compression (PMD)* <https://github.com/apasarkar/localmd>`_
- `*maskNMF initialization method for segmenting dense (spatially overlapping) calcium imaging data* <https://github.com/apasarkar/masknmf>`_
- `*localNMF demixing algorithm for demixing neural signals* <https://github.com/apasarkar/masknmf>`_
- `*Napari plugin for local fast video rendering of all results* <https://github.com/apasarkar/napari-masknmf>`_
    

Sharp Edges: 

- **What input data is supported?** Right now, users can only pass a multipage tiff file as input. 
- **Can I analyze multiple datasets in the same session?** No, the app is designed for interactive analysis of a single dataset at a time. 
