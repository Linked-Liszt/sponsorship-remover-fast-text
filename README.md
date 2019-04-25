# sponsorship_remover_fastText
A FastText prototype for sponsoff



## Requirements 

Python 3

#### Libraries installable through pip:
```
numpy
youtube_transcript_api
pandas
```

#### Installing FastText

Direct link to official guide: https://github.com/facebookresearch/fastText/tree/master/python

#### Excerpt from guide:
Using Pip. 
```
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ pip install .
```


## How to use

All files have constants at the top in UPPER_SNAKE_CASE. File pointers and parameters should easily be modifiable from there. 

### A quick walkthrough

Make sure data.csv is placed in /data. Data is in format: ```text, sentiment``` (0 is sponsored content, 1 is not)
The dataset included is from: https://github.com/Sponsoff/sponsorship_remover

#### 1. preprocessData.py
data.csv takes the .csv values and splits them into Test/Train files. 
The Test/Train ratio can be set by TEST_TRAIN_RATIO

```
>python .\preprocessData.py
>Processed 1343 rows: 403 test and 940 train.
```

#### 2. train.py
train.py is used to build the model. With HYPERPARAM_SEARCH set to false, it will train 1 model with the given parameters. 
With HYPERPARAM_SEARCH set to true it will iterate through hyperparamters defined by the constants. The best model will be saved.
```
> python  .\train.py
>Begin Hyperparamter Search with 1000 epochserML\fastTextProd.1
>Training Model with 0.1 learning rate and 1 ngrams
...
```

#### 3. evaluateVideo.py
evaluate.py is designed to be run from command line easily. It has 3 parameters
```
--i [id]-> youtube video id (example msjuRoZ0Vu8)
--v -> add this to see verbose output of every evaluation
--p -> add this to see performance. displays performance of loading and run time 
```

```
> python .\evaluateVideo.py --i msjuRoZ0Vu8 --p
>[{'start': '0:00:00', 'end': '0:00:14'}, {'start': '0:11:37', 'end': '0:11:48'}, {'start': '0:11:48', 'end': '0:11:58'}, {'start': '0:12:20', 'end': '0:12:33'}]
>Total Time: 0.03399538993835449 (s)
>Model Load Time: 0.031005859375 (s)
>Model Run Time: 0.002989530563354492 (s)
```
