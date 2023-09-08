.. maskNMF post-processing guide

Post-processing
===============

The ultimate outcome of the analysis is an RESULTS.npz file, accessible through a download button located at the bottom of the app. This RESULTS.npz file encompasses both the PMD compressed data and the most recent demixing results generated through the interactive demixing procedure (provided you have executed the demixing step).

We have designed a napari plugin (see the Summary page) which contains a custom reader and viewer for visualizing all the data present in the RESULTS.npz file on your local computer. 