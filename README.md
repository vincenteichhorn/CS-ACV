# Assignments of Applied Hands-on Computer Vision (ACV) course for Master CS at HPI

## Assignments
1. [Assignment 01a: Data Curation and Augmentation for Image Classification](src/assignment01a.ipynb)
2. [Assignment 01b: MNIST 11-Class Classifier for Problematic Samples Detection](src/assignment01b.ipynb)

## Setup
```
poetry install
source $(poetry env info --path)/bin/activate
```

### Additional Setup for Assignment 01b
Either generate the curated dataset by running the notebook for Assignment 01a, extract the provided `mnist-curated.zip` (see [here](data/mnist-curated.zip)) into the `data/` folder named `mnist-curated`, or download the curated dataset from [🤗 Hugging Face](https://huggingface.co/datasets/vincenteichhorn/mnist-curated) and extract it into the `data/` folder.
