Personality Recognition using Open text dataset
===========================================================

abs.

## 1. Description

- ubuntu 16.04 or later
- python 3.5


## 2. Requirements

```
sudo apt-get install python3-pyyaml
sudo apt-get install python3-tk
sudo pip3 install -r requiremets.txt
rosrun feature_handler nltk_download.py
python3 -m spacy download en

```

## 3. Usage

```
rosrun feature_handler text_normalizer.py
rosrun model_interfacer model_interfacer.py 
```


## 4. Downloard personality dataset
```
mkdir -p dataset
curl -L -o ./dataset/mypersonality.csv https://raw.githubusercontent.com/hyeonukbhin/myPersonality-dataset/master/mypersonality.csv

curl -L -o ./dataset/mypersonality.csv https://raw.githubusercontent.com/hyeonukbhin/myPersonality-dataset/master/mypersonality.csv

```

