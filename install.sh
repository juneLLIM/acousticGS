# 1. Create conda environment
conda env create -f environment.yml -y

# 2. Activate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate acousticGS

# 3. Install submodules
pip install -e ./submodules/diff-gaussian-rasterization --no-build-isolation