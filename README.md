# A federated graph learning method to multi-party collaboration for molecular discovery

This is an implementation for Federated Lanczos Graph (FedLG).



## Table of Contents

- [Table of Contents](#table-of-contents)
- [Setup](#setup)
- [Datasets](#datasets)
  - [MoleculeNet](#moleculenet-httpsmoleculenetorgdatasets-1)
  - [LITPCBA](#litpcba-httpsgithubcomidruglabfp-gnnblobmaindatarar)
  - [DrugBank](#drugbank-httpsgithubcomkexinhuang12345castertreemasterddedata)
  - [BIOSNAP](#biosnap-httpsgithubcomkexinhuang12345castertreemasterddedata)
  - [CoCrystal](#cocrystal-httpsgithubcomsaoge123ccgnettreemaindata)
- [Preprocess](#preprocess)
- [Usage](#usage)

## Setup

To install the conda virtual environment `FedLG`:
```shell script
$ bash setup.sh
```
We use CUDA Version 11.7. If you have a different version of CUDA, please ensure that you install the appropriate versions of PyTorch and the CUDA toolkit that are compatible with your CUDA setup. Additionally, make sure that the versions of PyTorch and PyTorch Geometric are mutually compatible. For a straightforward installation of PyTorch Geometric, refer to the [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#).

## Datasets
Download data to the `FedLG/dataset` folder.

#### MoleculeNet: https://moleculenet.org/datasets-1.

#### LITPCBA: https://github.com/idrugLab/FP-GNN/blob/main/Data.rar.

#### DrugBank: https://github.com/kexinhuang12345/CASTER/tree/master/DDE/data.

#### BIOSNAP: https://github.com/kexinhuang12345/CASTER/tree/master/DDE/data.

#### CoCrystal: https://github.com/Saoge123/ccgnet/tree/main/data.

## Preprocess
If run the `main.py` directly, it may take a long time to preprocess dataset, so first run:
```shell script
python dataloaer.py --root MoleculeNet --dataset tox21 -- split smi --seed 4567
```
All parameters of dataloader:
```
usage: dataloader.py [--root] [--dataset] [--split] [--seed] [--device]

optional arguments:
  --root           --root directory for differernt molecular discovery databases: 
                     MoleculeNet, DrugBank, BIOSNAP, LITPCBA, CoCrystal

  --dataset        --In different root directory, choose dataset of different databases:
                     MoleculeNet: bbbp, MoleculeNet: bace, MoleculeNet: sider, MoleculeNet: tox21,
                     MoleculeNet: toxcast, MoleculeNet: esol, MoleculeNet: lipo, MoleculeNet: freesolv,
                     LIT-PCBA: ALDH1, LIT-PCBA: FEN1, LIT-PCBA: GBA, LIT-PCBA: KAT2A,
                     LIT-PCBA: MAPK1, LIT-PCBA: PKM2, LIT-PCBA: VDR,
                     DrugBank: DrugBank, CoCrystal: CoCrystal, BIOSNAP: BIOSNAP

  --split          --split type for different root and dataset:
                     smi, smi1, smi2

  --seed           --fixed data initialization and training seeds

  --device         --cuda or cpu
```

## Usage
```shell script
CUDA_VISIBLE_DEVICES=${your_gpu_id} python main.py --save_dir 'results' --alg fedlg --model MPNN --split smi --global_round 100 --local_round 5  --root MoleculeNet --dataset tox21  -- split smi --seed 4567 
```
All parameters of main:
```
usage: main.py [--alg] [--root] [--dataset] [--node_size] [--bond_size] [--hidden_size] [--extend_dim] [--output_size] [--model] [--split] [dropout] [--message_steps] [--num_clients] [--alpha] [--null_value] [--seed] 
               [--weight_decay] [--eps] [constant] [--delta] [--dp] [--batch_size] [--device] [--save_dir] [--beta1] [--beta2] [--local_round] [--proj_dims] [--lanczos_iter] [--global_round] [--comm_optimization] [--lr] [--clip]

optional arguments:
  --alg               --federated learning algorithm:
                        fedavg, fedprox, fedsgd, fedlg, fedadam, fedchem

  --root              --root directory for differernt molecular discovery databases: 
                        MoleculeNet, DrugBank, BIOSNAP, LITPCBA, CoCrystal

  --dataset           --In different root directory, choose dataset of different databases:
                        MoleculeNet: bbbp, MoleculeNet: bace, MoleculeNet: sider, MoleculeNet: tox21,
                        MoleculeNet: toxcast, MoleculeNet: esol, MoleculeNet: lipo, MoleculeNet: freesolv,
                        LIT-PCBA: ALDH1, LIT-PCBA: FEN1, LIT-PCBA: GBA, LIT-PCBA: KAT2A,
                        LIT-PCBA: MAPK1, LIT-PCBA: PKM2, LIT-PCBA: VDR,
                        DrugBank: DrugBank, CoCrystal: CoCrystal, BIOSNAP: BIOSNAP

  --node_size         --molecular node size

  --bond_size         --molecular bond size

  --hidden_size         hidden size

  --extend_dim        --extend dim for neural network

  --output_size       --output size

  --model             --graph neural network:
                        MPNN, GCN, GAT

  --split             --split type for different root and dataset:
                        smi, smi1, smi2

  --drooput           --dropout rate

  --message steps     --message step for graph neural network

  --num_clients       --clients number, here we set the max clients number is up to 4

  --alpha             --alpha for molecule dirichlet distribution

  --null_value        --null value

  --seed              --fixed data initialization and training seed

  --weight_decay      --weight decay for optimizer

  --eps               --epsilons distribution

  --constant          --constant for local differently privacy

  --delta             --differential privacy parameter

  --dp                --if True, use differential privacy

  --batch_size        --batch size of the model training:
                        32, 64 or 128

  --device            --cuda or cpu

  --save_dir          --results save directory, the model test results is saved to ./results/

  --beta1             --beta1 for Adam optimizer

  --beta2             --beta2 for Adam optimizer

  --local_round       --local model training round

  --proj_dims         --project dim of lanczos algorithm

  --lanczos_iter      --the iterations of lanczos

  --global_round      --global model training round

  --comm_optimization --using Bayesian Optimization or not

  --lr                --the learning rate of graph model

  --clip              --clip value for local differently privacy
```
