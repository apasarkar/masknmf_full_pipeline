.. maskNMF demixing guide

Running a demixing analysis using maskNMF
=========================================

As soon as the Preprocessing (motion correction and compression) steps have completed, the next stage is to idenify ROIs and demix them from the PMD compressed data. Conceptually, this stage can be broken into two steps: ROI identification and signal demixing. Through the Dash app, you can quickly and interactively go back and forth between these two stages in a "multi-pass" strategy. This allows you to initialize one set of cells, unmix these signals, and then repeat the process on the "residual" data (the data formed by subtracting off the cells you have already initialized from the PMD compressed representation of the movie). 

**Step 1: ROI Identification**

There are two initializations options you can use: (1) You can use the superpixels approach first defined in Buchanan et al. (which is a local correlation based algorithm for identifying active cells) or the maskNMF cell initialization pipeline, called "Dense Data" in the app. This was designed with for denser, volumetric imaging in mind but also works well for sparse, soma-targeted data. For the superpixels approach, the app provides a correlation threshold which you can interactively toggle to see how different correlation values affect the neural signal initialization. Finally, note that the "Dense Data" initialization is meant to be used for the first pass through the data; all subsequent passes use superpixel initialization.

**Step 2: Demixing**

After completing the cell initialization, click "run demixing" to kick off the localNMF demixing algorithm. Once this step is complete, you can inspect the results in interactive summary plots (see below). 

**Multipass Strategy**

The expected workflow is to first run Step 1 and Step 2. Then, you can use the visualization tools within the Dashboard to run subsequent passes of the algorithm. Note that subsequent passes will be via the superpixel initialization. 

**Interactive Summary Plots**

The first panel to view demixing results shows the shape of all cells which have been identified. If you click any pixel of that data, a summary of the demixing for that pixel will appear, showing the PMD trace, the neuropil estimate, each of the individual signals which are there, and more. 

The second panel provides an option to scroll through each ROI individually and examine its estimated temporal trace. Note that the ordering of the ROIs here corresponds to the ROIs from the first panel. 
