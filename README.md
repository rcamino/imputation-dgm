# Imputation with Deep Generative Models

This is the code used for the paper [Improving Missing Value Imputation with Deep Generative Models](https://arxiv.org/abs/1902.10666).

I submitted this paper to ICML 2019 and got rejected.
This is a work in progress, but some people asked me to release the code.
I have a lot to improve and I am pretty much alone coding this, so if you have suggestions or if you find problems
I will appreciate the feedback.

## Requirements

This code was tested with Python 3.6.7.
All the python libraries required to run this code are listed on the file `requirements.txt`.
I recommend installing everything with pip inside a virtual environment.
Run this code inside the project directory root:

```bash
# create the virtual environment with python 3
virtualenv --python=/usr/bin/python3 venv

# activate the virtual environment
. venv/bin/activate

# install the requirements with pip
pip install -r requirements.txt

# just in case to have the project on the python path
echo `pwd` > venv/lib/python3.6/site-packages/imputation-dgm.pth
```

If CUDA is available the code will detect it automatically, but a different PyTorch package might need to be installed.

## How to run

There are several scripts to run different parts of the project:

* Download, encode and scale the datasets:
  * Breast Cancer: `imputation_dgm/pre_processing/breast/download_and_transform.py`
  * Default Credit Card: `imputation_dgm/pre_processing/default_credit_card/download_and_transform.py`
  * Letter Recognition: `imputation_dgm/pre_processing/letter_recognition/download_and_transform.py`
  * Online News Popularity: `imputation_dgm/pre_processing/online_news_popularity/download_and_transform.py`
  * Spambase: `imputation_dgm/pre_processing/spambase/download_and_transform.py`
* Split the datasets into train and test: `imputation_dgm/pre_processing/train_test_split.py`
* Train GAIN: `imputation_dgm/methods/gain/trainer.py`
* Train VAE: `imputation_dgm/methods/vae/trainer.py`
* Train VAE-iterative:
  * First train VAE
  * Then run: `imputation_dgm/methods/vae/iterative_imputation.py`
  * The same model and hyperparameters must be used on both steps!
* Train VAE-backprop: same as VAE-iterative but adding the argument `--noise_learning_rate`

All the split variants can be run passing the argument `--temperature` (for the gumbel-softmax).
  
Every executable script has a commandline interface that will print a descriptive help when passing the argument `-h`.

 ## Example run
 
 In the script `run.sh` you will find an example for the pre-processing and for every method.
 
 You will need writing permission on the directory executing the scripts.
 
 If you execute the script entirely as it is, it may take **a long time**.

Also, take into account that the script does not use the same hyperparameters I used, so you might get different results.

## About the GAIN implementation

I based my implementation on the author's [code example](https://github.com/jsyoon0823/GAIN).

I could not reproduce the GAIN paper results exactly so I exchanged some questions with one of the authors [here](https://github.com/jsyoon0823/GAIN/issues/2)
and via email (I thank [jsyoon0823](https://github.com/jsyoon0823) for his patience).
Other people also asked more questions [here](https://github.com/jsyoon0823/GAIN/issues).

Take into account that my implementation has some "optional architecture changes"
based on my paper (the multi-input and multi-output).