# masknmf_full_pipeline
End to end pipeline for maskNMF, from motion correction to demixing

# Deployment Methods
We provide a Dash app for interacting with our calcium and voltage imaging pipelines

## Local Deployment: 
Use the command

'''
INSERT COMMAND TO LAUNCH DOCKERFILE HERE
'''

This command takes two inputs: (1) The tagged dockerfile to run (here, masknmf_full_pipeline) and (2) The dataset to analyze, which should be located in the "datasets" folder.

TODO: Eliminate dependency here on "datasets" folder


## Server deployment



## Lightning AI deployment
Coming soon


## General notes (to formalize soon)
Make sure pip install dash but also pip install "dash[diskcache]" so that you can run the app. Also pip install dash-extensions and flask-caching (pip install Flask-Caching)