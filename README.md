# PoetryT5
Deep Learning &amp; Digital Humanities (WS 21/22): Poetry generation with ByT5

## Create environment:
> conda create --name poetryT5 python=3.9  
> conda activate poetryT5  
> conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch   
> pip install -r requirements.txt  

## Install
First install the project using the setup.py:
> python setup.py install

Afterwards create the dataset and preprocess it via custom commands:
> pt5-dataset  
> pt5-preprocess

## Train
Start the training of the model via:
> pt5-train

## Testing
You can generate samples for testing using our inference.py script and setting the example path to the path to your trained model. Then run:  
> python poetryT5/inference.py

We also provide example scripts for computing the metrics:
> python poetryT5/evaluate.py  
> python poetryT5/max_sim.py
> python poetryT5/perplexity.py  
## Baseline
The code for our baseline, that we trained on colab, can be found in the  poetryT5/train_gpt2.ipynb

