# Spacecraft Crash Predictor

This project aims to build a classifier that predicts potential spacecraft crashes in advance, based on the time series of the subject's control and status in a pre-defined window. 

This work is part of a project accepted by NASA/TRISH's GoForLaunch program, with a greater aim to enhance human performance in space flights and postural balancing tasks. For details, see [Dr. Vimal's project description and research roadmap](https://sites.google.com/view/vivekanandpandeyvimal/research_2/current-research?authuser=0#h.on5o56f1ne5m). The project is the collaboration between the Neuroscience and CS departments at Brandeis University. 

# Requirements

## General Package Requirement

Among other required packages, we crucially need to make sure the following versions are correct:

- tensorflow 2.2.0 (```$ pip install tensorflow==2.2.0```)
- keras 2.4.3 (```$ pip install keras==2.4.3```)
- numpy 1.19.2 (```$ pip install numpy==1.19.2```)

### Notes to installing on HPCC

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

## Dataset Requirement

In order to run the code properly, you'll need to put under [data/](data/) directory the raw data file ```data_all.csv```, which contains all the raw data under columns listed in ```ESSENTIAL_RAW_COLS``` in [consts.py](src/consts.py).

# Run CV Training Experiments

```./src/experiment.py``` runs an experiment pipeline, which puts together preprocessing, modeling, evaluation, and logging. 

Under project root, run ```$ ./src/experiment.py -h``` to see all available arguments. 

- Example 1: running the all default options

  ```
  $ ./src/experiment.py
  ```

- Example 2: template for tweaking most of the parameters

  ```
  $ ./src/experiment.py --window 2.0 --ahead 1.0 --rolling 0.7 --configID 1 --early_stop \
  						   --model gru --crash_ratio 1 --dropout 0.5 --hidden_dim 100 \
  						   --batch_size 256 --notes "initial test run, diff in window, gru"
  ```

- Example 3: if working on an interactive shell, add ```--pbar``` to show progress bar

  ```
  $ ./src/experiment.py --pbar <other arguments>
  ```

# Evaluate Saved Model on Held-out Test Sets
With [predict.py](src/predict.py), the model obtained from running [expeirment.py](src/experiment.py) above can be evaluated on test sets from datasets, potentially with a different look ahead time. 

## What you'll need
Before running [predict.py](src/predict.py), you'll need to have run at least 1 experiment where you've saved a model. In this way, the script can read the training record and saved model under [exp/](exp) folder. 

You'll need to specify the id of the experiment (found under ```Experiment ID``` column in ```results/exp_ID_config.csv```). By default, if no ```--ahead``` time is specified, the model will evaluate on the original dataset (which mean the original time ahead).

## Example
For example, if we want to evaluate the model saved from Experiment 206 (i.e. ```Experiment ID```=206), use

```
$ .src/predict.py 206
```

If you'd like to evaluate the model from Experiment 206 on a test set of a different time ahead, e.g. 0.7s (700ms) ahead, specify that time with:
```
$ .src/predict.py 206 --ahead 0.7
```
Note that if the dataset with the new time ahead does not exist, the program will automatically generate the dataset before evaluating the model.

# Examine Results

## Result Files

Under [results/](results/), ```experiment.py``` script automatically saves the the experiment configuration (in ```exp_ID_config.csv```) and result metrics (in ```exp_results_all.csv```). If these files don't exist upon recording, they will be created from the templates in ```./reasults/template```. 

Each experiment will be indexed with a unique ID number. The two tables can be joined on their first columns, i.e. on ```experiment ID``` or ```exp_ID```. ```exp_ID_config.csv``` contains configuration information such as:

- preprocessing configurations (e.g. window size, time ahead, rolling step, etc.)
- model configurations (e.g. input data configuration, model type, model hyperparameters, batch size, etc.)
- training configurations (e.g. whether and which cross validation strategy to use, early stopping, etc.)
- display configurations (e.g. whether to show progress bar for preprocessing and training, experiment notes, etc.)

```exp_results_all.csv``` contains test set results information such as:

- confusion matrix on test set data
- various metrics on test set data, such as AUC, F1, accuracy, precision, recall, etc.
- if a CV strategy is used, standard deviation and mean of the metrics will be reported.

## Merging Result Files

The following command allows you to join ```exp_results_all.csv``` (left) and  ```exp_ID_config.csv``` (right) on ```experiment ID``` column and save the combined table to ```hpcc_results.csv```. 

```
$ ./src/summary.py --merge
```

To avoid overwriting existing merged files, An integer will be appended at the end of the output file

# Dev Tips: How to Add New Code

## Code for Extracting New Feature
Whenever a new feature is proposed:
1. Write its extraction function in [extract_features.py](src/processing/extract_features.py)
2. In [consts.py](src/consts.py), add the feature name and column path to ```COL_PATHs``` dictionary
3. If a new, non-categorical feature has a range much larger than joystick's [-1, 1], e.g. position and velocity: this feature is a large value feature, and you should add its name to ```LARGE_VAL_FEATS``` in [consts.py](src/consts.py). In this way, the large features will be normalized to match joystick's range if a ```--normalize large``` option is specified.
4. If the feature is categorical, e.g. destabilizing joystick deflection, add it to ```CATEGORICAL_FEATS``` in [consts.py](src/consts.py), since we crucially do NOT want to normalize them in any normalization mode.
5. Add feature extraction function call as ```self._save_col(<function call and return>, <feature name>)``` in ```__init__``` method of [MARSDataLoader](src/processing/marsdataloader.py)
6. When you initialize a new MARSDataLoader, it will automatically include the new feature.

## Code for New Feature Combination (i.e. Configuration of Datasets used in Training)
Whenever a new feature combination is proposed:
1. Pick a unique intger ```n``` as your ```config_id```, e.g. ```100```, and put it in set ```CONFIG_IDS``` in [consts.py](src/consts.py)

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

## Code for New Model Architecture
Whenever a new model build has been proposed:

- For ```tensorflow.keras.Sequential``` models (see ```LSTM``` and ```GRU``` for examples):

  1. Create a new build function to return a complied model in the approporiate script in ```scr/modeling/```

  2. Add the new model's name to [consts.py](src/consts.py), with all caps variable name and lowercased string name, i.e. ```MODEL_NAME="name"```, and insert it into list ```AVAILABLE_MODELS``` and another model subgroup (e.g. ```RNN_MODELS```) as you see fit.

  3. Insert the following script into ```match_and_build_model``` function in [experiment.py](src/experiment.py):

     ```
     elif args.model in C.<MODEL_SUBGROUP>:
     		model = <call build function>
     ```

  Then, when you run ```experiment.py``` with ```--model <new model name>```, the new model will be used.

- Other models: TBD! Note: if not a Keras model, saving methods such as ```model.save()``` may not be compatible. 

## Code for New Experiment Option

Whenever you need to add a new option to the experiment:

1. add the option to ```argparser``` in ```main()``` in [experiment.py](src/experiment.py)
2. if you'd like the argument to have a different display name in the [experiment configuration file](results/template/exp_ID_config.csv), add a ```('<option name>', '<name to display in results file>')``` to the ```COL_CONV``` dictionary in [consts.py](src/consts.py), 
3. new settings by default gets appended after the template columns. To have the new option appear in a specific place, add it to the template, and the change will be reflected after the experiment output has been cleared.

Note: only numbers, booleans, np.float, and strings will be saved. See ```ACCEPTABLE_TYPES``` in [consts.py](src/consts.py).

## Clearing Experiment Output

The code has been organized in a way that additional experiment options and evaluation metrics will be appended to the existing configuration file and results file as new columns, and missing columns will simply be left blank. Thus, there's no need to clear output for new code added.  

However, if you'd still like to (e.g. due to bugs found in preprocessing code or model output), clear the following directories:

- every visible file and directory under [data](data/) except for ```data_all.csv```
- every visible file and directory under [results/](results/) except for the [template/](results/template) folder
- every visible file and directory under [exp/](exp/)

# Acknowledgements

Special thanks to [Jie Tang](https://github.com/TJmask), some of whose [code logic for the same task](https://github.com/TJmask/Space-Health-Predicting) has been especially helpful in building this project; Jie's name will be included the docstring sections of the functions, wherever credit is due. 
