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

   ```shell script
   conda create -n tf2
   conda activate tf2
   ```

2. Install anaconda python 3.8 if not already

   ```shell script
   conda install -c anaconda python=3.8
   ```

3. Install the packages listed above in the given order:

   ```shell script
   pip install tensorflow==2.2.0
   pip install keras==2.4.3
   pip install numpy==1.19.2
   pip install --user scipy matplotlib ipython jupyter pandas tqdm sklearn
   ```

> Note to **Apple M1 chip** users: after following the instructions above, ```import tensorflow``` on Apple M1 chip may result in ```40087 illegal hardware instruction  python``` and Python quitting, in which case please:
> 1. follow [this solution on SO](https://stackoverflow.com/a/68214296/6716783) to install the compatible version of TensorFlow, where we'll need to run the following steps:
>
>    > Step 3 Install the tensorflow wheel called `tensorflow-2.4.1-py3-none-any.whl` located at this public google drive link https://drive.google.com/drive/folders/1oSipZLnoeQB0Awz8U68KYeCPsULy_dQ7
>
>    > Step 3.1 Assuming you simply installed the wheel to downloads run `pip install ~/Downloads/tensorflow-2.4.1-py3-none-any.whl` in your activated virtual environment
>
>    > Step 4 Type python which will bring up `>>>`in your terminal and type
>    >
>    > ```py
>    > >>> import tensorflow
>    > >>>
>    > ```
>    >
>    > If there is no 'zsh illegal hardware instruction" error you should be good to go.
>
>    Note that it's okay to have Tensorflow 2.4.1 after implementing this solution; we did not face any incompatibility issues with other packages such as keras and numpy. 
>
> 2. make sure all ```import keras``` are replaced with ```from tensorflow import keras``` to prevent import errors, according to [this answer](https://stackoverflow.com/a/67019039/6716783). 

## Dataset Requirement

In order to run the code properly, you'll need the raw data file ```data_all.csv```, which contains at least all the columns listed in ```ESSENTIAL_RAW_COLS``` in [consts.py](src/consts.py).

### Instructions(Dataset)

1. Download the data from [this Google Drive link](https://drive.google.com/drive/folders/1vKRGvyq-vcICy5moTjkdT1lydyQ4myxQ?usp=sharing) (please contact repo owner for access). 
2. Go to your desired ```dataset_name``` folder and download data.
3. Rename the downloaded file to ```data_all.csv``` if not already. 
4. In the [data/](data/) directory, create a new folder called ```dataset_name``` and put the file under [data/dataset_name] directory. After this step, the file should exist at  [data/dataset_name/data_all.csv](./data/dataset_name/data_all.csv).
>Note that `SupineMARS` dataset will be used in this demonstration. You can also use other dataset names, as long as their data file exists at `data/<dataset_name>/data_all.csv`.

### Instructions(Changes in [consts.py](src/consts.py).)
When added new dataset, you'll need to make changes in [consts.py](src/consts.py). in order to run the code properly. 

1. Open [consts.py](src/consts.py), below ```SUPINE_MARS = "SupineMARS"```, you should add new code as: 


```shell script
   [DATASET_NAME] = " DATASET_NAME "
```
2. Update the current ```DATA_SUBDIR_LIST = [SUPINE_MARS, UPRIGHT_MARS]``` to

```shell script
   DATA_SUBDIR_LIST = [SUPINE_MARS, UPRIGHT_MARS,DATASET_NAME]
```

# Reproduce the Best Model

In our experiment, we chose the stacked GRU model trained on 1000ms (1s) window size and 800ms time-in-advance (0.8s) as our best model for further result analysis. To train this model, use the commands describe in this section in the given order.

Note that `SupineMARS` dataset will be used in this demonstration. You can also use other dataset names, as long as their data file exists at `data/<dataset name>/data_all.csv`.

### Step 1. Generate the dataset and train the model 

using the following command:

```shell script
./src/experiment.py SupineMARS --window 1.0 --ahead 0.8 \
                   	--early_stop --cv_mode kfold --cv_splits 10 \
                   	--save_model \
                   	--configID 3 --normalize all --model gru-gru \
                   	--notes "stacked gru: converge on AUC"
```

where:

* `SupineMARS`: name of the dataset used for training
* ```--configID 3``` : use feature set containing {angular position: float, angular degree: float, joystick deflection: float, destabilizing joystick deflection: bool}
* ```--normalize all```: normalize all non-categorical features, in this case: angular position, angular degree, joystick deflection.
* ```--model gru-gru```: train a double-layer stacked GRU model; if you'd just like to test out the entire pipeline, change ```gru-gru``` to ```mlp``` to save time. 
* ```--save_model```: save the trained model for prediction. 

After this process is done, you should expect the following files to be generated:

* a folder containing preprocessed dataset for this window size and time-in-advance (i.e. ```ahead```) combination, named ```1000window_800ahead_200rolling```, will be created in [data/SupineMARS](./data/SupineMARS) among other auxiliary files. 
* The best performing model during the 10-fold CV for this experiment saved at [exp/](./exp) folder, named similar to ```exp1_SupineMARS_1000win_800ahead_conf1_gru-gru```, including the tensorflow model file and a customized experiment recorder object. ```exp1``` indicates that the experiment ID is 1. 
* ```exp_ID_config.csv``` and ```exp_results_all.csv``` under [results/](./results/) folder, the former with the configuration of all the experiments and the latter with CV mean and std for various evaluation metrics, both indexed by experiment ID. 

#### But I just wanna run a quick test!

OK, try this 1 epoch, no CV, command that trains a linear model: 

```shell script
./src/experiment.py SupineMARS --pbar --window 1.0 --ahead 0.8 --max_epoch 1 --cv_mode disable --save_model --configID 1 --model linear --notes "testing only: linear no CV"
```

Note that if you don't have a preprocessed ```--window 1.0 --ahead 0.8``` dataset, it will take ~20-40min to create the dataset, depending on your hardware. 

### Step 2. Merge CV results

To view ```exp_ID_config.csv``` and ```exp_results_all.csv```  in one file, merge them by running: 

```shell script
./src/summary.py --merge
```

Follow the standard output to locate the merged file, which, if no duplicate file has been generated before, should be [results/hpcc_results.csv](./results/hpcc_results.csv), also indexed by experiment ID.

### Step 3. Evaluate the Model on Test Set

To evaluate the model on the test set and generate test set predictions for analysis, run:

```shell script
./src/predict.py SupineMARS <YOUR EXP NUMBER> --save_preds_csv --save_lookahead_windows --at_recall 0.95
```

where:

* `SupineMARS`: name of the dataset used for test; note that it does NOT have to be the same as the training test, e.g. you can train on `SupineMARS` and test on `UprightMARS`
* ```<YOUR EXP NUMBER>``` corresponds to the exp# for the model saved at [exp/](./exp) folder. If this is your first time running this program, it should be ```1```. 
* ```--save_preds_csv```: save all the test set input data windows and their predictions
* ```--save_lookahead_windows```: save the features in the time-in-advance duration corresponding to each input sliding window, i.e. between ```endtime_of_sliding_window``` and ```endtime_of_sliding_window + time_in_advance_duration```. We use these features to perform error analysis, e.g. savable crashes.
* ```--at_recall 0.95```: when generating testset predictions, set the decision threshold of the model to the value that yields a recall of 0.95. 

After this process is done, you should expect the following files to be generated:

* [results/heldout_pred_res.csv](./results/heldout_pred_res.csv): test set evaluation metrics for all experiments; each test set prediction has its own ID because each exp ID could be evaluated multiple time using different precision at recall thresholds. 
* a table named ```<pred_id>heldoutPred_expID<expID>_<threshold>DecThresh_FalsePreds_1000.0win_800.0trainahead_0testahead.csv``` under [results/](./results/), containing the **False** predicitons in the test dataset and a table named ```<pred_id>heldoutPred_expID<expID>_<threshold>DecThresh_TruePreds_1000.0win_800.0trainahead_0testahead.csv``` under [results/](./results/), containing the **True** predicitons in the test dataset. We primarily used these two files in our error analysis. 

# Detailed Explanation for Each Step

You can skip this section if you only wish to recreate the best model, which you should be able to obtain by following the previous section. 

As above, we'll use `SupineMARS` dataset for demonstration purposes, but you can choose any dataset, as long as it follows [Dataset Requirement](#dataset-requirement).

## Run CV Training Experiments

```./src/experiment.py``` runs an experiment pipeline, which puts together preprocessing, modeling, evaluation, and logging. 

Under project root, run ```$ ./src/experiment.py -h``` to see all available arguments.

- Example 1: running the all default options

  ```shell script
  ./src/experiment.py SupineMARS
  ```

- Example 2: template for tweaking most of the parameters; note that the last 3 lines are usually held the same during the experiments, and the last 2 lines are default values that can be left out of the arguments.

  ```shell script
  ./src/experiment.py SupineMARS --window 1.0 --ahead 0.8 --rolling 0.2 \
                      --model gru --normalize all --crash_ratio 1 --configID 3 \
                      --notes "gru-def: +destablize, +norm all, gru save_model" \
                      --early_stop --save_model \
                      --patience 3 --cv_splits 10 --cv_mode kfold \
                      --batch_size 256 --dropout 0.5 --hidden_dim 100 --threshold 0.5 --max_epoch 50
                      
  ```

- Example 3: if working on an interactive shell, add ```--pbar``` to show progress bar

  ```shell script
  ./src/experiment.py SupineMARS --pbar <other arguments>
  ```

## Evaluate Saved Model on Held-out Test Sets
With [predict.py](src/predict.py), the model obtained from running [experiment.py](src/experiment.py) above can be evaluated on test sets from datasets, potentially with a different look ahead time. 

### What you'll need
Before running [predict.py](src/predict.py), you'll need to have run at least 1 experiment where you've saved a model. In this way, the script can read the training record and saved model under [exp/](exp) folder. 

You'll need to specify the id of the experiment (found under ```Experiment ID``` column in ```results/exp_ID_config.csv```). By default, if no ```--ahead``` time is specified, the model will evaluate on the original dataset (which mean the original time ahead).

### Example
For example, if we want to evaluate the model saved from Experiment 206 (i.e. ```Experiment ID```=206) and save the held-out test set predictions as csv files, use

```shell script
./src/predict.py SupineMARS 206 --save_preds_csv
```

To evaluate the models at a different recall such as 0.99 (i.e. evaluate with the decision threshold that yields 0.99 recall), use

```shell script
./src/predict.py SupineMARS 206 --save_preds_csv --at_recall 0.99
```

If you'd like to evaluate the model from Experiment 206 on a test set of a different time ahead for testing model rigidity, e.g. 0.7s (700ms) ahead, specify that time with:
```shell script
$ .src/predict.py SupineMARS 206 --ahead 0.7
```
Note that if the dataset with the new time ahead does not exist, the program will automatically generate the dataset before evaluating the model.

## Examine Results

### Result Files

Under [results/](results/), ```experiment.py``` script automatically saves the the experiment configuration (in ```exp_ID_config.csv```) and result metrics (in ```exp_results_all.csv```). If these files don't exist upon recording, they will be created from the templates in ```./reasults/template```. 

Each experiment will be indexed with a unique ID number. The two tables can be joined on their first columns, i.e. on ```experiment ID``` or ```exp_ID```. See below section for automatically joining the tables. 

```exp_ID_config.csv``` contains configuration information such as:

- preprocessing configurations (e.g. window size, time ahead, rolling step, etc.)
- model configurations (e.g. input data configuration, model type, model hyperparameters, batch size, etc.)
- training configurations (e.g. whether and which cross validation strategy to use, early stopping, etc.)
- display configurations (e.g. whether to show progress bar for preprocessing and training, experiment notes, etc.)

```exp_results_all.csv``` contains test set results information such as:

- confusion matrix on test set data
- various metrics on test set data, such as AUC, F1, accuracy, precision, recall, etc.
- if a CV strategy is used, standard deviation and mean of the metrics will be reported.

### Merging Result Files

The following command allows you to join ```exp_results_all.csv``` (left) and  ```exp_ID_config.csv``` (right) on ```experiment ID``` column and save the combined table to ```hpcc_results.csv```. 


```bash
./src/summary.py --merge
```

To avoid overwriting existing merged files, an integer will be appended at the end of the output file

### Printing Dataset Statistics

To summarize dataset stats (e.g. total samples, train-test split sizes, crash-noncrash ratios, etc.) of a specific dataset, e.g. SupineMARS, use `--dataset_path` and specify the path to dataset directory:

```shell
./src/summary.py --dataset_path data/SupineMARS
```

Then, you can find a new file at [data/SupineMARS/dataset_stats.csv](data/SupineMARS/dataset_stats.csv) containing all the stats.

# HPCC Users: Automated Experiments

**NOTE: this section is only for people with an HPCC account AND access to the lab's shared Google Drive**

To reduce the time needed to run multiple experiment on HPCC using sbatch, we've introduced a new [submit_hpcc_experiment.py](src/submit_hpcc_experiment.py) script which reads the experiment configurations specified in [hpcc_exp_configs.csv](hpcc_exp_configs.csv), one row per experiment, and automatically submit the corresponding experiment via sbatch.

## Required Files

* [submit_hpcc_experiment.py](src/submit_hpcc_experiment.py): main script, in repo already, no action required
* [hpcc_exp_configs.csv](hpcc_exp_configs.csv): experiment configurations, one row per experiment. ```job_name```, ```gpu_type```, ```program_path``` and ```notes``` are obligatory columns. The rest of the columns correspond to the arguments one would like to pass to [experiment.py](./src/experiment.py) when running experiments
  * Unrecognized arguments would cause the job to submit but fail very soon ([experiment.py](./src/experiment.py) does not recognize the arguments). 
  * Unspecified arguments will be the default value
  * Boolean options (e.g. ```--save_model```) will have TRUE or FALSE in the configs table. TRUE means the option will be passed to experiment.py and FALSE means not. 
* sbatch_template.txt: template for creating each sbatch experiment. Here's how to obtain it: 
  * Download [this file on our shared Google Drive Folder](https://drive.google.com/file/d/1XNKvnItNIgYCHmyhN-GSZkJAu29VhuES/view?usp=sharing)
  * Put the file under [tmp/](tmp/) folder.
  * Change the email address for ```mail-user``` in line 9 and save
  * **NOTE: this file need to be downloaded and edited manually since .gitignore will automatically exclude the file 

## Instructions

### Step 1. Edit [hpcc_exp_configs.csv](hpcc_exp_configs.csv) with the parameters desired

**Make sure you have downloaded all the requied file and place them at the required directories. 

Now, you may refer back to Step 1 in [Reproduce the best model](https://github.com/yonglin-wang/spacecraft-crash-predictor/tree/exp_auto_readme#reproduce-the-best-model) for specific parameter input. 

Each row in ```.csv ``` indicates one experinment:
* If you would like to run multiple experiments at once, you should append the parameter settings for the next experiment after the previous one(s). 
* If you would like to run new experiments, you should replace the previous parameter settings. 

### Step 2. Move to the right directory 

Using the following command:
  ```shell script
  cd <path_to_project_root>
  ```
NOTE: Please make sure you are running the following in the project root directory (e.g. $WORK/spacecraft-crash-predictor/ if you cloned this repo to your work directory on HPCC).

### Step 3. Run the automated experiments

Using the following command:
  ```shell script
  python src/submit_hpcc_experiment.py
  ```
You should see the terminal output looking like this:
  ```shell script
  =*-+=*-+=*-+=*-+=*-+Now processing experiment at row 8 (0-based index)...=*-+=*-+=*-+=*-+=*-+
   formatted arguments: --window 1.5 --ahead 1.0 --configID 3 --model linear --crash_ratio 1 --normalize all --early_stop --save_model
   echoing formatted args... --window 1.5 --ahead 1.0 --configID 3 --model linear --crash_ratio 1 --normalize all --early_stop --save_model
   Command above finished with exit code
  ```
However, if you receive output like the following:
   ```shell script
   sh: sbatch: command not found
   Traceback (most recent call last):
     File "/Users/mac/git/spacecraft-crash-predictor-1/src/submit_hpcc_experiment.py", line 144, in <module>
       main()
     File "/Users/mac/git/spacecraft-crash-predictor-1/src/submit_hpcc_experiment.py", line 44, in main
       raise ValueError(f"exit code {exit_code} is not 0! exp_info_dict content: {exp_info_dict}")
   ValueError: exit code 32512 is not 0! exp_info_dict content: OrderedDict([('job_name', 'sl-test'), ('gpu_type', 'TitanXP'), ('program_path', './src/experiment.py')
   ```
this indicates that you may have entered the parameters incorrectly (e.g. sbatch not installed, wrong parameter names, invalid parameter values, etc). Please go back to the [hpcc_exp_configs.csv](hpcc_exp_configs.csv) and double check your configurations. 

Otherwise, if you are able to receive the final message, like the following:
   ```shell script
   [Final Message] Congrats! Successfully submitted 9 jobs!
   ```
Then, you are good to go!

# Dev Tips
For tips on adding new models and new features, etc., please see [src folder's README](https://github.com/yonglin-wang/spacecraft-crash-predictor/blob/main/src/README.md)
