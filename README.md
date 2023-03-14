# masknmf_full_pipeline
End to end pipeline for maskNMF, from motion correction to demixing

# Deployment Methods
We provide a Dash app for interacting with our calcium and voltage imaging pipelines


## Local use for end users:
To launch the app, place your dataset in the folder "datasets" and run:

```
make launch dataname=<YOUR_DATASET_NAME>
```

This will initiate the app. Then follow the instructions displayed on your command line. In particular, you will be instructed to go to localhost:X where X is a port number (for example localhost:8901). This is port to which you need to connect on the system on which the app is hosted. If this is your local system, simply type "localhost:X" into your browser and the app will be launched.

## Local use for developers:
To build a dockerfile, navigate to the top level directory of this repository and run:

```
make build
```


## Cloud deployment
Coming soon
