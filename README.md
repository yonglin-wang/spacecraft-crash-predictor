# Spacecraft Crash Predictor

This project aims to build a classifier that predicts spacecraft crashes in advance. Special thanks to Jie Tang, some of whose [code logic for the same task](https://github.com/TJmask/Space-Health-Predicting) has been especially helpful in building this project; Jie's name will be included the docstring sections of the functions, wherever credit is due. 

# Requirements

## General Package Requirement

Among other required packages, we crucially need to make sure the following versions are correct:

- tensorflow 2.2.0
- keras 2.4.3 (```$ pip install keras==2.4.3```)
- numpy 1.19.2

## Notes to installing on HPCC

These are the exact steps I used to install the exact versions listed above:

1. Create new environment named, e.g. tf2, and activate it: 

   ```
   $ conda create -n tf2
   $ conda activate tf2
   ```

2. Install anaconda python 3.8 if not already

   ```
   $ conda install -c anaconda python=3.8
   ```

3. Install the packages listed above in the given order:

   ```
   $ pip install tensorflow==2.2.0
   $ pip install keras==2.4.3
   $ pip install numpy==1.19.2
   ```

# How to Add New Code

## Code for Extracting New Feature
Whenever a new feature is proposed:
1. Write its extraction function in [extract_features.py](src/processing/extract_features.py)
2. Add feature name and column path to ```COL_PATHs``` dictionary in [consts.py](src/consts.py)
3. Add feature extraction function call as ```self._save_col(<function call and return>, <feature name>)``` in ```__init__``` method of [MARSDataLoader](src/processing/marsdataloader.py)
4. When you initialize a new MARSDataLoader, it will automatically include the new feature.

## Code for New Feature Combination (i.e. Configuration of Datasets used in Training)
Whenever a new feature combination is proposed:
1. Pick a unique intger ```n ``` as your ```config_id```, e.g. ```100```, and put it in set ```CONFIG_IDS``` in [consts.py](src/consts.py)

2. Write the configuration function with name in format similar to ```_generate_config_<n>``` in [dataset_config.py](src/processing/dataset_config.py), which takes at least a ```MARSDataLoader``` object and returns a 6-tuple of numpy arrays ```train_inds```, ```test_inds```, ```X_train```, ```X_test```, ```y_train```, ```y_test```

3. Insert ID and configuration function in ```load_splits``` in [dataset_config.py](src/processing/dataset_config.py), with following logic:
    ```
    elif config_id == <config_id>:
        config = <name of new config function>
    ```
    
4. Insert an entry with the following format in dictionary ```CONFIG_SPECS``` in [consts.py](src/consts.py)

    ```
    <config_id>: {COLS_USED: [], CONFIG_OVERVIEW: ""}
    ```

Then, when you create a new instance of ```MARSDataLoader``` and call ```load_splits``` in [dataset_config.py](src/processing/dataset_config.py) with ```config_id=<new config ID>```, the 6-tuple with the new configuration will be returned.

## Code for New Model 
Whenever a new model build has been proposed:

- For ```tensorflow.keras.Sequential``` models (see ```LSTM``` and ```GRU``` for examples):

  1. Create a new build function to return a complied model in the approporiate script in ```scr/modeling/```

  2. Add the new model's name to [consts.py](src/consts.py), with all caps variable name and lowercased string name, i.e. ```MODEL_NAME="name"```, and insert it into list ```AVAILABLE_MODELS``` and another model subgroup (e.g. ```RNN_MODELS```) as you see fit.

  3. Insert the following script into [experiment.py](src/experiment.py) where a build function is called:

     ```
     elif args.model in C.<MODEL_SUBGROUP>:
     		model = <call build function>
     ```

  Then, when you run ```experiment.py``` with ```--model <new model name>```, the new model will be used.

- Other models: TBD!