.. maskNMF pre-processing guide

Motion correction, Compression, Inspection
==========================================

In this section, we focus on the steps to preprocess data in the maskNMF app. See <image> for how the app looks at the beginning.

The first step is to select the file to analyze. If you are launching this app through neurocaas or through the official docker container, that means you really selected the data when you launched the app - in this case, will be in the "mounted_data" folder. You can use the file selection bar to browse through the app filesystem and select it.

The next step is to motion correct and compress the data. This step is optional and can be skipped if, for example, the data has already been registered or if there is no motion due to experimental settings. If you wish to run without motion correction, under Step 2 simply turn off the switch for "registration". 

If, however, you want to run motion correction, keep this switch turned out. In this case, you will have to decide whether you would like to save the motion corrected movie. If you do, you will be able to compare the motion corrected data with the compressed data. This is a useful validation metric. However, this will involve explicitly saving out the movie, resulting in a duplication of data. For time-constrained analyses (adaptive experiments), you can elect to register *and* compress the data "on the fly". 

