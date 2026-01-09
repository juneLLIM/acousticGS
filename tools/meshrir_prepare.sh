data="~/data/dataset/MeshRIR/S1-M3969"
base=$(pwd)
mkdir -p $data
cd $data
wget https://zenodo.org/records/10852693/files/S1-M3969_npy.zip
unzip S1-M3969_npy.zip
cd $base
python tools/meshrir_split.py --base_folder $data