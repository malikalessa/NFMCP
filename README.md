# NFMCP


### The repository contains code refered to the work:

# Striving for Simplicity in Deep Neural Models Trained for Malware Detection
Malik AL-Essa, Giuseppina Andresini, Annalisa Appice, Donato Malerba

# Cite this paper

The paper has been accepted in New Frontiers in Mining Complex Patterns Workshop,ECML PKDD 2023 Conference.



### Code Requirements

 * [Python 3.9](https://www.python.org/downloads/release/python-390/)
 * [Keras 2.7](https://github.com/keras-team/keras)
 * [Tensorflow 2.7](https://www.tensorflow.org/)
 * [Scikit learn](https://scikit-learn.org/stable/)
 * [Matplotlib 3.5](https://matplotlib.org/)
 * [Pandas 1.3.5](https://pandas.pydata.org/)
 * [Numpy 1.19.3](https://numpy.org/)
 * [Dalex 1.4.1](https://github.com/ModelOriented/DALEX)
 * [adversarial-robustness-toolbox 1.9](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
 * [scikit-learn-extra 0.2.0](https://scikit-learn-extra.readthedocs.io/en/stable/)
 * [Hyperopt 0.2.5](https://pypi.org/project/hyperopt/)


###  Description for this repository
Two different types of datasets are used in this work CICMaldroi20, and CICMalMem22. MinMax scaler has been used to normalize the datasets. The datasets and models that have been used in work can be downloaded through [Datasets and Models]()
  
  
   

### How to use

The implementation for all the experiments used in this work are listed in this repository.
  * main.py : to run the implementation
 


## Replicate the Experiments
* ###### defensive = 0   &emsp;        # 1 for defensive phase, 0 for offensive phase <br />

* ###### TRAIN_BASELINE = 0   &emsp;        # 1 train baseline with hyperopt <br />
* ###### Dalex = 1   &emsp; # 1 to generate features ranking based on Dalex  <br />
* ###### Rank_Dalex_features_Train = 0 &emsp;  # 1 To compute feature selection using Dalex Features<br />
 
* ###### MI_Train = 1   &emsp; # 1 To compute feature selection using MI Features  <br />

* ###### Attack_MI = 1      &emsp;          #  1 To attack the model using  the ranking of the training dataset <br />

