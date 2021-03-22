wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh -b -p $HOME/anaconda3
./anaconda3/bin/conda init bash
. ~/.bashrc
conda env create -f environment.yml
conda activate spinningup