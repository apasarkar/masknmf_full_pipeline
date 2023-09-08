.. maskNMF summary

Summary
=======
This is a `Dash app <https://github.com/apasarkar/masknmf_full_pipeline>` for GPU-accelerated interactive data analysis of functional neuro-imaging data using the maskNMF pipeline. Through the app you can motion correct (non-rigid), compress (PMD), and demix (NMF variant) imaging data. We provide convenient visualization options at each of these stages along the way. You can export the compressed and demixed results easily in a format for local viewing (eg Napari) and further analysis.

The app brings the following individual data analysis methods into one cohesive pipeline for fast interactive data analysis. 


- `*Accelerated NoRMCoRRE* <https://github.com/apasarkar/jnormcorre>`
- `*Accelerated SVD-like data compression (PMD)* <https://github.com/apasarkar/localmd>`
- `*maskNMF initialization method for segmenting dense (spatially overlapping) calcium imaging data* <https://github.com/apasarkar/masknmf>`
- `*localNMF demixing algorithm for demixing neural signals* <https://github.com/apasarkar/masknmf>`
- `*Napari plugin for local fast video rendering of all results* <https://github.com/apasarkar/napari-masknmf>`
    
