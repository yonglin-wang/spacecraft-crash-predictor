# Dev Tips: How to Add New Code
## Running Experiments on HPCC
Research assisstants at Brandeis should refer to [Initial Set Up Instructions](https://docs.google.com/document/d/1s9GzIPajZSfqRGhotvU8dn_GdYl3CIp_b3plCz54Xq0/edit?usp=sharing) for setting up HPCC environment from scratch (i.e. all the way from creating account to using sbatch to run the test commands). 

Please contact the repo owner (YW) if you need access to the document.

## Code for Extracting New Feature
Whenever a new feature is proposed:
1. Write its extraction function in [extract_features.py](processing/extract_features.py)
2. In [consts.py](consts.py), add the feature name and column path to ```COL_PATHs``` dictionary
3. If a new, non-categorical feature has a range much larger than joystick's [-1, 1], e.g. position and velocity: this feature is a large value feature, and you should add its name to ```LARGE_VAL_FEATS``` in [consts.py](consts.py). In this way, the large features will be normalized to match joystick's range if a ```--normalize large``` option is specified.
4. If the feature is categorical, e.g. destabilizing joystick deflection, add it to ```CATEGORICAL_FEATS``` in [consts.py](consts.py), since we crucially do NOT want to normalize them in any normalization mode.
5. Add feature extraction function call as ```self.___save_new_feature(<function call and return>, <feature name>)``` in the if statement in ```__init__``` method of [MARSDataLoader](processing/marsdataloader.py)
6. When you initialize a new MARSDataLoader, it will automatically scan and load the new feature; if the new feature file is not found, it will create it using the extraction call as specified in Step 5.

## Code for New Feature Combination (i.e. Configuration of Datasets used in Training)
Whenever a new feature combination is proposed:
1. Pick a unique integer ```n``` as your ```config_id```, e.g. ```3```, and put it in set ```CONFIG_IDS``` in [consts.py](consts.py)

2. Write the configuration function with name in format similar to ```_generate_config_<n>``` in [dataset_config.py](processing/dataset_config.py), which takes at least a ```MARSDataLoader``` object and returns a 6-tuple of numpy arrays ```train_inds```, ```test_inds```, ```X_train```, ```X_test```, ```y_train```, ```y_test```

3. Insert ID and configuration function in ```load_splits``` in [dataset_config.py](processing/dataset_config.py), with following logic:
    ```
    elif config_id == <config_id>:
        config = <name of new config function>
    ```
    
4. Insert an entry with the following format in dictionary ```CONFIG_SPECS``` in [consts.py](consts.py)

    ```
    <config_id>: {COLS_USED: [], CONFIG_OVERVIEW: ""}
    ```

Then, when you create a new instance of ```MARSDataLoader``` and call ```load_splits``` in [dataset_config.py](processing/dataset_config.py) with ```config_id=<new config ID>```, the 6-tuple with the new configuration will be returned.

## Code for New Model Architecture
Whenever a new model build has been proposed:

- For ```tensorflow.keras.Sequential``` models (see ```LSTM``` and ```GRU``` for examples):

  1. Create a new build function to return a complied model in the approporiate script in ```scr/modeling/```

  2. Add the new model's name to [consts.py](consts.py), with all caps variable name and lowercased string name, i.e. ```MODEL_NAME="model_name"``` (e.g. ```GRU = "gru"```), and insert it into list ```AVAILABLE_MODELS``` and another model subgroup (e.g. ```RNN_MODELS```) as you see fit (to be used in if statements in ```match_and_build_model``` below).

  3. Then, in [experiment.py](experiment.py):
     1. ```import``` the new build function
     2. Insert the following script into ```match_and_build_model``` function 

        ```
        elif model_name in C.<MODEL_SUBGROUP>:
                model = <call build function>
        ```

  Then, when you run ```experiment.py``` with ```--model <new model name>```, the new model will be used.

- Other models types: TBD! Note: if not a Keras model, saving methods such as ```model.save()``` may not be compatible. 

## Code for New Experiment Option

Whenever you need to add a new option to the experiment:

1. add the option to ```argparser``` in ```main()``` in [experiment.py](experiment.py)
2. if you'd like the argument to have a different display name in the [experiment configuration file](../results/template/exp_ID_config.csv), add a ```('<option name>', '<name to display in results file>')``` to the ```COL_CONV``` dictionary in [consts.py](consts.py), 
3. new settings by default gets appended after the template columns. To have the new option appear in a specific place, add it to the template, and the change will be reflected after the experiment output has been cleared.

Note: only numbers, booleans, np.float, and strings will be saved. See ```ACCEPTABLE_TYPES``` in [consts.py](consts.py).

## Clearing Experiment Output

The code has been organized in a way that additional experiment options and evaluation metrics will be appended to the existing configuration file and results file as new columns, and missing columns will simply be left blank. Thus, there's no need to clear output for new code added.  

However, if you'd still like to (e.g. due to bugs found in preprocessing code or model output), clear the following directories:

- every visible file and directory under [data](../data/) except for ```data_all.csv```
- every visible file and directory under [results/](../results/) except for the [template/](../results/template) folder
- every visible file and directory under [exp/](../exp/)
