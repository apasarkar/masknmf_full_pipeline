# ----- Settings ------

# Use Single Bash Shell For All Commands
.ONESHELL:
SHELL := /bin/bash
# ----- Constants -----

# Set Project Root For Absolute Paths 
PROOT=$(shell pwd)
#  ----- Docker Management Recipes -----

# Fetch image from dockerhub (if not already downloaded)
fetch:	
	sudo docker pull apasarkar/masknmf_full_pipeline

# Build Main & Tag Build-Image (If new build-image was built)
build:	
	sudo docker build --tag apasarkar/masknmf_full_pipeline .
    
publish:	
	sudo docker push apasarkar/masknmf_full_pipeline

# Test Docker Image
launch: fetch
	echo "Go to localhost:8981 on server to access app"    
	docker run -it -p 8981:8900 --gpus=all\
  		--mount type=bind,source=$(PROOT)/datasets/$(dataname),destination=/mounted_data/$(dataname) apasarkar/masknmf_full_pipeline