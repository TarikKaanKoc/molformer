# MoLFormer

1. [Getting Started](#getting-started)
    1. [Pretrained Models and training logs](#pretrained-models-and-training-logs)
    2. [Replicating Conda Environment](#replicating-conda-environment)
3. [Data](#data)
    1. [PreTraining Datasets](#pretraining-datasets)
    2. [Finetunning Datasets](#finetunning-datasets)
5. [PreTraining](#pretraining)
6. [FineTunning](#finetunning)
7. [Attention Visualization Analysis](#attention-visualization-analysis)


## Getting Started

**This Code and Environment have been tested on Nvidia V100s**

#### Pretrained Models and training logs
The Pretrained models and associated training logs are located in the /data directory in the following hierarchy. 

```
data/
├── checkpoints
|   └── linear_model.ckpt
|   └── full_model.ckpt
├── Full_Attention_Rotary_Training_Logs
│   ├── events.out.tfevents.1628698179.cccxc544.604661.0
│   └── hparams.yaml
└── Linear_Rotary_Training_Logs
    ├── events.out.tfevents.1620915522.cccxc406.63025.0
    └── hparams.yaml
```

#### Replicating Conda Environment 

Due to the use of apex.optimizers in our code, Apex must be compiled from source. Step-by-step directions are provided in [environment.md](environment.md)

## Data

### PreTraining Datasets
The code expects zinc15 and pubchem datasets to be located in ```data/``` directory
  * Zinc15 should be in located ```data/ZINC/``` in and expected in multiple smi files with an smiles string per line.
  * PubChem should be located in ```data/pubchem/ and expects a single “CID-SMILES” text file with 2 columns (index and smiles string).
    We took the raw Pubchem dataset and converted every smiles molecule into the canonical form, utilizing rdkit, as well as trimmed down the file itself. 
    Our dataloader expects Pubchem to be in our converted form and will not run on the raw pubchem file. 
### Finetunning Datasets
The code expects the finetunning datasets to be in the following hierarchy. These datasets wete provided in the finetune_datasets.zip

```
data/
├── bace
│   ├── test.csv
│   ├── train.csv
│   └── valid.csv
├── bbbp
│   ├── test.csv
│   ├── train.csv
│   └── valid.csv
├── clintox
│   ├── test.csv
│   ├── train.csv
│   └── valid.csv
├── esol
│   ├── test.csv
│   ├── train.csv
│   └── valid.csv
├── freesolv
│   ├── test.csv
│   ├── train.csv
│   └── valid.csv
├── hiv
│   ├── test.csv
│   ├── train.csv
│   └── valid.csv
├── lipo
│   ├── lipo_test.csv
│   ├── lipo_train.csv
│   └── lipo_valid.csv
├── qm9
│   ├── qm9.csv
│   ├── qm9_test.csv
│   ├── qm9_train.csv
│   └── qm9_valid.csv
├── sider
│   ├── test.csv
│   ├── train.csv
│   └── valid.csv
└── tox21
    ├── test.csv
    ├── tox21.csv
    ├── train.csv
    └── valid.csv
```


## PreTraining
To train a model from Scratch. 

#### PreReqs
    1. The code expects an datasets as described in [PreTraining Datasets](pretraining_datasets)
    2. The code expects an environmnet as described in [environment.md](environment.md)

#### Running Training tasks
    1. Activate Conda Environment
    2. ```cd training```
    3. ```bash run_pubchem_light.sh```

## FineTunning

### PreReqs
  1. The code expects an datasets as described in [Finetunning Datasets](finetunning_datasets)
  2. The code expects an environmnet as described in [environment.md](environment.md)

#### Running Finetune Tasks
To Finetune the pretrained model:

  1. Activate Conda Environment
  2. ```cd finetune```
  3. ```bash run_finetune_mu.sh```

#### Finetune Results

During Finetuning a diretory is created named ```checkpoint_<measure_name>```. 
The path to the results csv will be in the form of ```./checkpoint_<measure_name>/<measure_name>/results/results_.csv```
The ```results_.csv``` file contains 4 columns of data. Column one contains the validation score for each epoch while column 2 contains the test 
score for each epoch. Column 3 contains the best validation score observed up to that point of fine tuning while column 4 is the test score of 
the epoch which had the best validation score.

## Attention Visualization Analysis
The `notebooks` directory provide attention visualization for two setup with Rotary Embeddings:
- **Linear attention** (./notebooks/full_attention_rotary/attention_analysis_rotary_full.ipynb)
- **Full attention** (./notebooks/linear_attention_rotary/attention_analysis_rotary_linear.ipynb)

The checkpoints required for the above models are to be placed in `./data/checkpoints`
