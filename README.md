This is an implementation of our paper in COLING2020:

**One Comment from One Perspective: An Effective Strategy for Enhancing Automatic Music Comment**



This code is based on our baseline MMPMS: https://github.com/gyhd/python_study/tree/c84ee2e945bcf86a1066360bbf4be774836444ff/paddle_models-develop/PaddleNLP/Research/IJCAI2019-MMPMS



### Requirements
- Python == 3.7
- PaddlePaddle == 1.8.1
- NLTK == 3.4.5
- numpy == 1.18.1
- pandas == 1.0.3

### Folder structure
```
/dataset                 : two datasets constructed by us
/data/ES_newdata/  	     : data path
/models  				 : models design
/output 				 : logs and results
run.py 					 : train or referece
eval.py 	   		     : eval the result of inference
```


### Prepare data

The vocabulary and the preprocessed data will be saved in the folder:

```
data/ES_newdata/
├── music.train                need to prepare
├── music.test					 need to prepare
├── music.valid						 need to prepare
├── music.train.pkl				
├── music.valid.pkl
├── music.test.pkl
├── addcomments_small_embedding.txt        embedding file, need to prepare: https://ai.tencent.com/ailab/nlp/en/index.html
└── vocab.json
```

Preprocess the data by running preprocess.py :

```shell
python preprocess.py
```


### Train

To train a model, run:
```shell
python run.py 
```
Not use the distinction:

```shell
python run.py --fLoss_mode no
```

Use gpus:

```shell
python run.py --use_gpu True
```



### Inference

```shell
python run.py --infer --model_dir MODEL_DIR --result_file RESULT_FILE
```



### Evaluation

```shell
python eval.py RESULT_FILE
```



