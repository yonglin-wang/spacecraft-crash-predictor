# Space-Health-Predicting

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
2. Write the configuration function with name in format similar to ```_generate_config_<n>``` in [dataset_config.py](src/processing/dataset_config.py), which returns a 6-tuple of numpy arrays ```train_inds```, ```test_inds```, ```X_train```, ```X_test```, ```y_train```, ```y_test```
3. Add ID and configuration function in ```load_splits``` in [dataset_config.py](src/processing/dataset_config.py), with following logic:
    ```
    elif config_id == <config_id>:
        config = <name of new config function>
    ```
4. When you call ```load_splits``` in [dataset_config.py](src/processing/dataset_config.py) with the new config ID, the 6-tuple with the new configuration will be returned.

## Code for New Model 
TBD!