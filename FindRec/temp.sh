# # Remove current environment
# conda deactivate
# conda env remove -n mamba

# # Create new environment with Python 3.11
# conda create -n mamba python=3.11 -y
# conda activate mamba

# # Install PyTorch 2.4 with CUDA 12.1
# pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# # Verify versions
# python -c "import torch, sys; print(f'Python: {sys.version_info.major}.{sys.version_info.minor}'); print(f'PyTorch: {torch.__version__}')"

# # Install the wheels
# pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu122torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# pip install https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu122torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# # Test
# python -c "from mamba_ssm import Mamba; print('Success!')"

# pip install colorlog
# pip install tensorboard
# pip install texttable
# pip install colorama
# pip install pandas
# pip install scikit-learn
# pip install "numpy<2"


# download dataset
cd /data/
mkdir MicroLens-100k
cd MicroLens-100k
wget https://recsys.westlake.edu.cn/MicroLens-100k-Dataset/extracted_modality_features/MicroLens-100k_image_features_CLIPRN50.npy
wget https://recsys.westlake.edu.cn/MicroLens-100k-Dataset/extracted_modality_features/MicroLens-100k_title_en_text_features_BgeM3.npy
wget https://recsys.westlake.edu.cn/MicroLens-100k-Dataset/MicroLens-100k_pairs.csv
cd /home/kbhau001/recom/FindRec

#chang 'dataset': '/data/MicroLens-100k/' in temp.sh 
#changed utils.logger.py  below

    # logfilename = "{}/{}-{}-{}.log".format(
    #     config["model"], config["model"], get_local_time(), md5
    # )

