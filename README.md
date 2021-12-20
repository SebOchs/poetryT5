# PoetryT5
Deep Learning &amp; Digital Humanities (WS 21/22): Poetry generation with ByT5

## Create environment:
> conda create --name poetryT5 python=3.9
> conda activate poetryT5
> conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
> pip install -r requirements.txt

## Install
First install the project using the setup.py:
> python setup.py install

Afterwards create the dataset and preprocess it via custom commands:
> pt5-dataset
> pt5-preprocess