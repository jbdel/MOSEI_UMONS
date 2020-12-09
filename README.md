<img src="https://acl2020.org/assets/images/logos/acl-logo.png" width=10% /> &nbsp;&nbsp; <img src="https://raw.githubusercontent.com/valohai/ml-logos/5127528b5baadb77a6ea4b999a47b4e86bf0f98b/pytorch.svg" width=25% /><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<b>ACL2020</b> <br/>

Pytorch implementation of the paper <b>"A Transformer-based joint-encoding for Emotion Recognition and Sentiment Analysis"</b><br/>
Challenge-HML [Best Paper Award](https://jbdel.github.io/img/HML_best_paper.pdf)

```
@inproceedings{delbrouck-etal-2020-transformer,
    title = "A Transformer-based joint-encoding for Emotion Recognition and Sentiment Analysis",
    author = "Delbrouck, Jean-Benoit  and
      Tits, No{\'e}  and
      Brousmiche, Mathilde  and
      Dupont, St{\'e}phane",
    booktitle = "Second Grand-Challenge and Workshop on Multimodal Language (Challenge-HML)",
    month = jul,
    year = "2020",
    address = "Seattle, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.challengehml-1.1",
    doi = "10.18653/v1/2020.challengehml-1.1",
    pages = "1--7"
}
```

#### Model

The model Model_AV is the module used for the UMONS solution to the MOSEI dataset using only linguistic and acoustic inputs.<br/>
Results can be replicated at the following Google Colab sheet: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ir00q2drUzJ6bwIoOLodPErS6NjleZG4?usp=sharing)


#### Environement

Create a 3.6 python environement with:
```
torch              1.2.0    
torchvision        0.4.0   
numpy              1.18.1    
```

We use GloVe vectors from space. This can be installed to your environement using the following commands :
```
wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz
pip install en_vectors_web_lg-2.1.0.tar.gz
```
#### Data

Download data from [here](https://drive.google.com/file/d/1tcVYIMcZdlDzGuJvnMtbMchKIK9ulW1P/view?usp=sharing).<br/>
Unzip the files into the 'data' folder<br/>
More informations about the data can be found in the 'data' folder<br/>

#### Training

To train a Model_AV model on the emotion labels, use the following command :

```
python main.py --model Model_LA --name mymodel --task emotion --multi_head 4 --ff_size 1024 --hidden_size  512 --layer 4 --batch_size 32 --lr_base 0.0001 --dropout_r 0.1
```
Checkpoints are created in folder `ckpt/mymodel`

Argument `task` can be set to `emotion` or `sentiment`. To make a binarized sentiment training (positive or negative), use `--task_binary True`

#### Evaluation 

You can evaluate a model by typing : 
```
python ensembling.py --name mymodel
```
The task settings are defined in the checkpoint state dict, so the evaluation will be carried on the dataset you trained your model on.

By default, the script globs all the training checkpoints inside the folder and ensembling will be performed.

#### Results:

Results are run on a single GeForce GTX 1080 Ti.<br>
Training performances:
| Modality                          |     Memory Usage  | GPU Usage  |  sec / epoch | Parameters | Checkpoint size | 
| ------------- |:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Linguistic + acoustic             | 320 Mb | 2400 MiB |  103 | ~ 33 M | 397 Mb
| Linguistic + acoustic + vision    |

You should approximate the following results :

| Task Accuracy  |     val | test | test ensemble | epochs | 
| ------------- |:-------------:|:-------------:|:-------------:|:-------------:|
| Sentiment-7    | 43.61   |  43.90  | 45.36  | 6      
| Sentiment-2    |  82.30  |  81.53  | 82.26  |  8        
| Emotion-6      | 81.21   |  81.29  | 81.48  |  3    

Ensemble results are of max 5 single models <br>
7-class and 2-class sentiment and emotion models have been train according to the instructions [here](https://github.com/A2Zadeh/CMU-MultimodalSDK/blob/master/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/README.md).<br>

#### Pre-trained checkpoints:
Result `Sentiment-7 ensemble` is obtained from these checkpoints : [Download Link](https://drive.google.com/file/d/11BKBbxp2tNZ6Ai1YD-pPrievffYh7orM/view?usp=sharing)<br/>
Result `Sentiment-2 ensemble` is obtained from these checkpoints : [Download Link](https://drive.google.com/file/d/15PanBXsxXzvmDsVuA5qiWQd33ssezjxn/view?usp=sharing)<br/>
Result `Emotion ensemble` is obtained from these checkpoints : [Download Link](https://drive.google.com/file/d/1GyXRWhtf0_sJQacy5wT8vHoynwHkMo79/view?usp=sharing)<br/>
