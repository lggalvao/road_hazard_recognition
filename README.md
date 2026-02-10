Setup Linux

Install Miniconda to create isolated env
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh

Create Conda Env
conda create -n pytorch_env python=3.11
conda activate pytorch_env

Install gdown to download data from google drive
pip install gdown

Download files from Google Drive 
gdown --fuzzy google link

Insttall unrar to extract .rar files
sudo apt install unrar

Extract .rar file
unrar x .rar_file_path path_to_extract

Clone the Code from git hub
Install Git
sudo apt-get install git

Setup github account
git config --global user.name "Name"  # Set your name for all repositories.
git config --global user.email "email@example.com"  # Set your email for all repositories.


