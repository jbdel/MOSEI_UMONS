## Multilogue-net - Official PyTorch Implementation  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multilogue-net-a-context-aware-rnn-for-multi/multimodal-sentiment-analysis-on-cmu-mosei-1)](https://paperswithcode.com/sota/multimodal-sentiment-analysis-on-cmu-mosei-1?p=multilogue-net-a-context-aware-rnn-for-multi) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multilogue-net-a-context-aware-rnn-for-multi-2/multimodal-sentiment-analysis-on-mosi)](https://paperswithcode.com/sota/multimodal-sentiment-analysis-on-mosi?p=multilogue-net-a-context-aware-rnn-for-multi-2)

------

Written by Hongyi:

To run the model, change the hyperparameters in the bash scripts `./train_xxxxxx.sh` if needed and execute the following commands with whatever tag you want (`lav` in this case):
```bash
./train_categorical.sh lav
./train_regression.sh lav
```
The best model will then be saved in `model/[categorical|regression]\_lav.model` and the final evaluation will be saved in `model/[categorical|regression]\_lav.result`.

For evaluation, run
```bash
python evaluate.py <task>_<tag>

[Example]
python evaluate.py categorical_lav
```

------

This repository contains the official implemention for the *ACL 2020* workshop [paper](https://www.aclweb.org/anthology/2020.challengehml-1.3/)  :
> **Multilogue-Net: A Context Aware RNN for Multi-modal Emotion Detection and Sentiment Analysis in Conversation**<br>
> Aman Shenoy and Ashish Sardana<br>
> https://arxiv.org/abs/2002.08267  
>
> **Abstract:** *Sentiment Analysis and Emotion Detection in conversation is key in several real-world applications, with an increase in modalities available aiding a better understanding of the underlying emotions. Multi-modal Emotion Detection and Sentiment Analysis can be particularly useful, as applications will be able to use specific subsets of available modalities, as per the available data. Current systems dealing with Multi-modal functionality fail to leverage and capture - the context of the conversation through all modalities, the dependency between the listener(s) and speaker emotional states, and the relevance and relationship between the available modalities. In this paper, we propose an end to end RNN architecture that attempts to take into account all the mentioned drawbacks. Our proposed model, at the time of writing, out-performs the state of the art on a benchmark dataset on a variety of accuracy and regression metrics.*  

| ![dialogue](https://github.com/amanshenoy/multilogue-net/blob/master/diagrams/dialogue.jpg) |
|:-------------------------------------------------------------------------------------------:|


***Note***: *Since this project was pursued during the course of a summer internship, we do not intend to actively maintain this repository, and the following can only be considered a basic model and training implementation.*

## Resources and Dependencies

The datasets used to train all the models were obtained and preprocessed using the CMU-Multimodal SDK which can be found [here](https://github.com/A2Zadeh/CMU-MultimodalSDK). The `data` folder in the repository contains pre-processed data for the CMU-MOSEI Dataset, whose details can be found [here](https://www.aclweb.org/anthology/P18-1208/). If the pre-processed `data` folder is to be downloaded, the repository must be cloned using `git-lfs`, due to the size of the files.

The repository contains files consisting of all relevant models, dataloaders, formatted data, and training scripts to be able to train the model. This repository is only a basic implementation and does not include the scripts for majority of the experiments performed in the paper. 

The models in the repositories can be trained on the following target variables -  

* Binary Sentiment labels  
* Emotion labels (One of 6 emotions)
* Regression outputs (Real valued range between -3 to +3)  
  
The repository also contains a `.txt` requirements file consisting of all dependancies required to be able to train and infer the model on any of the labels by running

    >> pip install -r requirements.txt

## Implementation and Training

The repository contains three training scripts as per the desired target variables.  

The scripts require python3.6 or above and can be run as

    >> python train_categorical --no-cuda=False --lr=1e-4 --l2=1e-5 --rec-dropout=0.1 --dropout=0.5 --batch-size=128 --epochs=50 --class-weight=True --log_dir='logs/mosei_categorical'
  
    >> python train_emotion --no-cuda=False --lr=1e-4 --l2=1e-5 --rec-dropout=0.1 --dropout=0.5 --batch-size=128 --epochs=50 --class-weight=True --emotion='happiness' --log_dir='logs/mosei_emotion'
  
    >> python train_regression --no-cuda=False --lr=1e-4 --l2=1e-4 --rec-dropout=0.1 --dropout=0.25 --batch-size=128 --epochs=100 --log_dir='logs/mosei_regression'
    
Depending on the kind of prediction desired. Instructions on how training or inference is to be run for other datasets can be found in the `data` directory. 

Due to issues with git-lfs and the data being lost locally, you can find the `data` directory of a previous commit [here](https://github.com/amanshenoy/multilogue-net/tree/5d6b6ff8b1a26cf0762d6c1ca3a99917e881bf26/data). 

## Experimentation and Results 

The model training takes roughly 15 seconds/epoch for `train_emotion.py` and `train_categorical.py` and 40 seconds/epoch for `train_regression.py` on CMU-MOSEI, on a single NVIDIA GV100. All models mentioned can infer on a test conversation in under 1 second on a single NVIDIA GV100. 

At the time of writing Multilogue-net achieves state-of-the-art performance on emotion recognition, binary sentiment prediction, and sentiment regression problems on the CMU-MOSEI dataset.

## Citation

The paper in the ACL anthology can be found [here](https://www.aclweb.org/anthology/2020.challengehml-1.3/) and do cite our work if it proved to be useful in anyway!

``` 
@inproceedings{shenoy-sardana-2020-multilogue,
    title = "Multilogue-Net: A Context-Aware {RNN} for Multi-modal Emotion Detection and Sentiment Analysis in Conversation",
    author = "Shenoy, Aman  and
      Sardana, Ashish",
    booktitle = "Second Grand-Challenge and Workshop on Multimodal Language (Challenge-HML)",
    month = jul,
    year = "2020",
    address = "Seattle, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.challengehml-1.3",
    doi = "10.18653/v1/2020.challengehml-1.3",
    pages = "19--28",
    abstract = "Sentiment Analysis and Emotion Detection in conversation is key in several real-world applications, with an increase in modalities available aiding a better understanding of the underlying emotions. Multi-modal Emotion Detection and Sentiment Analysis can be particularly useful, as applications will be able to use specific subsets of available modalities, as per the available data. Current systems dealing with Multi-modal functionality fail to leverage and capture - the context of the conversation through all modalities, the dependency between the listener(s) and speaker emotional states, and the relevance and relationship between the available modalities. In this paper, we propose an end to end RNN architecture that attempts to take into account all the mentioned drawbacks. Our proposed model, at the time of writing, out-performs the state of the art on a benchmark dataset on a variety of accuracy and regression metrics.",
}
```
