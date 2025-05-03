###Set up conda environment
Load up anaconda: ```module load anaconda3/2024.10```
Create an Anaconda environment: ```conda create -n hf python=3.10 -y```
Activate the environment: ```conda activate hf```
Install compatible pytorch: ```pip install torch==1.13.0+cu117.with.pypi.cudnn -f https://download.pytorch.org/whl/torch_stable.html```

###Install package dependencies 
Change library path: ```export LD_LIBRARY_PATH=/home/cl5587/.conda/envs/hf/lib/```
Install the requirements: ```pip install -r requirements.txt --no-deps```

Then run download_model.py on command line for offline access (make sure to change cache directory)
