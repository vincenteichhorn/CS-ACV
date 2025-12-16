# Assignments of Applied Hands-on Computer Vision (ACV) course (Computer Science Master, HPI)

## Assignments

### Assignment 01: MNIST Image Classification and Data Curation 

- [Assignment 01a: Data Curation for MNIST Image Classification](src/assignment01a.ipynb)
- [Assignment 01b: MNIST 11-Class Classifier for Problematic Samples Detection](src/assignment01b.ipynb)

### Assignment 02: Multi-Modal Learning

- [Assignment 02a: Multi-Modal Learning Dataset Exploration](src/assignment02/01_dataset_exploration.ipynb)
- [Assignment 02b: Multi-Modal Learning Fusion Comparison](src/assignment02/02_fusion_comparison.ipynb)
- [Assignment 02c: Multi-Modal Learning Ablation Study](src/assignment02/03_maxpool2d_ablation.ipynb)
- [Assignment 02d: Multi-Modal Learning with RGB and LiDAR Data](src/assignment02/04_final_assessment.ipynb)

## Setup

```
poetry install
source $(poetry env info --path)/bin/activate
```

### Additional Setup for Assignment 01b

Either generate the curated dataset by running the notebook for Assignment 01a, extract the provided `mnist-curated.zip` (see [here](data/mnist-curated.zip)) into the `data/` folder named `mnist-curated`, or download the curated dataset from [🤗 Hugging Face](https://huggingface.co/datasets/vincenteichhorn/mnist-curated) and extract it into the `data/` folder.
