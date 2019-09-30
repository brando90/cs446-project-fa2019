# cs466-project-fa2019

Gdrive with original data set: https://drive.google.com/drive/u/3/folders/19obIXaG31RXJ7Ft3kRqh86K_ozypfnsQ
slides: TODO

The split data, generated using the notebook `automl/predicting_performance/data_splitting/Data Splitting.ipynb`, can be found [here](https://drive.google.com/drive/folders/1oAzWjqAlT7xDyldq8_7AxsHwj6NyhObc?usp=sharing). The folder includes `mapping.csv`, which maps models in the split dataset back to the orignal for reference.

Set up
-----

To install the project go to automl folder and run

```
pip install -e .
```

that installs the code without having to re-pip install it everytime there is a change.

Meta Learners
-----

Go to:

```
/automl/predicting_performance/meta_learners_predictors
```

thats where you should put your code new baselines. There is a Chain LSTM baseline there under `chain_lstm.py`. The main function in that file serves as an example on how to use the code. The unit tests are also good places to look to see how the code works.

Feature choices
-----

There are many features to choose from the raw data to include for the Neural Nets to use. In the end I decided to use:

1) Architecture + Architecture Hyper Params
2) For features related to the Optimizer the first few epochs from the training and validation history
3) For features related to initialization the mean, std and l2 norm at the beginning of training
4) Final train error

and using that we try to predict

5) The Test Error

the reason for the choices are as follow:

1) The meta learner needs to know the architecture and its hyperparams
2) Instead of using the Optimizer name as a symbol (and its hyperparam names) I decided that using a small fraction of the train and validation history (in fact its using all the history...which we need to change). Check paper for previous work on this: https://arxiv.org/pdf/1806.09055.pdf . This is better features because its agnostic to the optimizer (which could in theory be implemented by an RNN). Thus, allowing the performance predictor meta learner to be applied to *any* iterative optimizer.
3) Similar to instead of using the symbolic name of the initializer, statistics at the initialization before training allows the meta learner be agnostic to the initialization algorithm.
4) Final train error is usually a good proxy to the final test error, specially if the train error is none-zero. 

TODO
-----

- make sure the dataloader/colalte function only gives a fraction of the history and not all the history
- see the performance of the first baseline, the Chain LSTM
- divide the data set in train, val, test. Right now the code uses all the data for all 3 (just to make sure the code runs)
- make sure the `to(device)` is used correctly so GPUs are used correctly.
- what to do with Nan errors if they are present in the train history

will make gitissue for these. Close them and update read me as you complete them
