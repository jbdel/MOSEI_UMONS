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
python main.py --model Model_bi --name mymodel --task emotion --multi_head 4 --ff_size 1024 --hidden_size  512 --layer 4 --batch_size 32 --lr_base 0.0001 --dropout_r 0.1
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

Results are run on a single GeForce GTX 1080 Ti.
You should approximate the following results :

| Task Accuracy  |     val | test | test ensemble | epochs | Memory usage |
| ------------- |:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Sentiment-7    | 43.66 |  44.46 | 45.51  |      | 2400MiB
| Sentiment-2    |       |        |        |      | 2400MiB
| Emotion        | 81.06 |  81.07 | 81.23  |      | 2400MiB

Ensemble results are of max 5 single models <br>
7-class and 2-class sentiment models have been train according to instructions [here](https://github.com/A2Zadeh/CMU-MultimodalSDK/blob/master/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/README.md).<br>

#### Pretrained checkpoints:

Result `Sentiment-7` is obtained from:

```
python main.py --seed 8206597 --model Model_bi_clean --name mymodel --task emotion --multi_head 4 --ff_size 1024 --hidden_size  512 --layer 4 --batch_size 32 --lr_base 0.0001 --dropout_r 0.1
```

Result `Sentiment-7 ensemble` is obtained from:
```
for seed in 8206597 3569479 2810648 9250778
do
  python main.py --seed ${seed} --model Model_bi_clean --name mymodel --task emotion --multi_head 4 --ff_size 1024 --hidden_size  512 --layer 4 --batch_size 32 --lr_base 0.0001 --dropout_r 0.1
done 
```

is available [here]()

For `Sentiment-2` obtained from:


For `Emotion` obtained from:





