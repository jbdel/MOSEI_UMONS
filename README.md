#### Model

The model-bi is the module used for the UMONS solution to the MOSEI dataset using only linguistic and acoustic inputs.

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

Download data from [here](https://drive.google.com/uc?id=130P2EJPKL_9bpjoXpYAAIgqVi62rHxVC&export=download) : <br/>
Unzip the files into the 'data' folder<br/>
More informations about the data can be found in the 'data' folder<br/>

#### Training

To train a model-bi model, use the following command :

```
python main.py --model model_bi --name mymodel --task emotion --multi_head 4 --ff_size 1024 --hidden_size  512 --layer 6 --batch_size 32 --lr_base 0.0001 --dropout_r 0.1
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

You should approximate the following results :

| Task Accuracy       val | test | test ensemble |
| ------------- |:-------------:|:-------------:|
| Sentiment-7    | 43.61 |  44.40 | 45.23|
| Sentiment-2    |       |    | |
| Emotion        | 81.06 |  81.07 | 81.23 |

Ensemble results are of max 5 single models <br>
7-class and 2-class sentiment models have been train according to instructions [here](https://github.com/A2Zadeh/CMU-MultimodalSDK/blob/master/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/README.md).<br>

#### Pretrained checkpoints:

For `Sentiment-7` obtained from:

```
python main.py --seed 6510310 --model model_bi --name glimpse_e_new --task emotion --multi_head 4 --ff_size 1024 --hidden_size  512 --layer 6 --batch_size 32 --lr_base 0.0001 --dropout_r 0.1
```
is available [here]()

For `Sentiment-2` obtained from:


For `Emotion` obtained from:





