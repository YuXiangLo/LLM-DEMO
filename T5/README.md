# This is the folder of Transformer (T5)

## Environment Setup
### Using Conda(Recommended)

1. `conda env create -f environment.yml`
2. `conda activate transformer`
3.  pip install -r requirements.txt

## Data Preprocessing

1.  remove quotes
2.  remove html tag
3.  remove LaTex tag

## T5 model

### Code Introduction

* Available model: t5-small, t5-base
    * t5-base has better performance
* Pytorch version: 11.8
* Custom Loss function: `loss = cross_entropy_error +  lambda * empty_string, lambda = 5`   
* Max\_length of each tokenized data: 512
* Batch\_size: The number of data that train at one iteration
* Epoch: How many times in total one data goes into the training
* gradient\_accumulation\_steps: update mdoel's parameters after this size of batches done training

### Output

* Cross entropy error
* Predicted string and the actual string(easy to compare by human)



