#!/bin/bash
set -eo pipefail
source /venv/bin/activate
#/opt/conda/bin/conda activate env
jupyter lab --ip 0.0.0.0 --no-browser --allow-root --port 8900 --NotebookApp.token='' --NotebookApp.password=''
