# masknmf_full_pipeline
End to end pipeline for maskNMF, from motion correction to demixing

# Deployment Methods
We provide a Dash app for interacting with our calcium and voltage imaging pipelines


## Local use for end users:
To launch the app, place your dataset in the folder "datasets" and run:

'''
make launch dataname=<YOUR_DATASET_NAME>
'''

This will launch 

## Local use for developers:
To build a dockerfile, navigate to the top level directory of this repository and run:

'''
make build
'''


## Cloud deployment
Coming soon