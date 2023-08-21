# masknmf_full_pipeline
End to end pipeline for maskNMF, from motion correction to demixing

# System requirements
So far, the app has been tested on Ubuntu systems (18.04, 20.04 and 22.04) with NVIDIA GPUs. We require the following linux/system tools: 

```
make
git
docker
nvidia-docker
```

# Deployment Methods
We provide a Dash app for interacting with our calcium and voltage imaging pipelines


## Local use for end users:
Start by cloning the repository: 

```
git clone https://github.com/apasarkar/masknmf_full_pipeline.git
```
To launch the app, place your dataset in the folder "datasets", navigate to the root directory of this repository ("cd masknmf_full_pipeline") and run:

```
make launch dataname=<YOUR_DATASET_NAME>
```

This will initiate the app. Then follow the instructions displayed on your command line. In particular, you will be instructed to go to localhost:X where X is a port number (for example localhost:8901). This is port to which you need to connect on the system on which the app is hosted. If this is your local system, simply type "localhost:X" into your browser and the app will be launched. If you are running this on the cloud or on a remote server, you will have to set up a SSH tunnel to this port to view things on your local machine. 

## Local use for developers:
Start by cloning the repository: 

```
git clone https://github.com/apasarkar/masknmf_full_pipeline.git
```

To build a dockerfile, navigate to the top level directory of this repository and run:

```
make build
```


## Cloud deployment
Coming soon
