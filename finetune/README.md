# Finetuning the pretrained for downstream tasks

Inorder to fine tune the saved model for various downstream tasks, run one of the scripts in this folder. For example, to train a classification model, run

```
conda activate molformer
./run_finetune_classification_tox21.sh
```

This code assumes that the pretrained model is in ../data/checkpoints/linear_model.ckpt

