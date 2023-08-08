conda create -y -n graph2smiles python=3.7 tqdm
conda activate graph2smiles
conda install pytorch==1.10.0 torchtext cudatoolkit=11.3 -c pytorch -c conda-forge
conda install -y pysocks
conda install -y rdkit -c conda-forge

# pip dependencies
pip install gdown OpenNMT-py==1.2.0 networkx==2.5 selfies==1.0.3
