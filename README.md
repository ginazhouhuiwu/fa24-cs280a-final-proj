This project is a snapshot of an ongoing research project on camera path generation. 

### 0. Install Nerfstudio dependencies
[Follow these instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html) up to and including "tinycudann" to install dependencies and create an environment

This project has this branch dependency: https://github.com/nerfstudio-project/nerfstudio/tree/gina/auto_camera_render. 
Please make sure you check out the correct branch.

### 1. Clone this repo
`git clone https://github.com/ginazhouhuiwu/fa24-cs280a-final-proj`

### 2. Install this repo as a python package
Navigate to this folder and run `python -m pip install -e .`

### 3. Train a NeRF
Follow the instructions under Nerfstudio to download data and train a NeRF.

### 4. Run Autocam Viewer
Navigate to the folder containing your trained model and run
`ns-viewer-autocam --load-config [config_file_name]`
