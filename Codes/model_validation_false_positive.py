import os             
import glob
import pandas as pd
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
from datetime import timedelta
# import matplotlib.pyplot as plt
# import warnings

from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import to_categorical

from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



# os.chdir('data')
##### get all the files that contain ".csv" #######
# all_path = glob.glob( '*/**.csv' )

#### read all the positive cases
crash_feature_label_300ms_500ms = pd.read_pickle('data/crash_feature_label_500ms_500ms_test')

### read all the negative casese
noncrash_feature_label_300ms_500ms = pd.read_pickle('data/noncrash_feature_label_500ms_500ms_test')

#### merge both positive and negative together
records = pd.concat([crash_feature_label_300ms_500ms, noncrash_feature_label_300ms_500ms])
#data_final = data_final[['features_cal_vel','features_org_vel','label']]
training_all = records.iloc[:, [0,1,3,4,5]]
testing_all = records.iloc[:,2]

train_X, test_X, train_y, test_y = train_test_split(training_all, testing_all, test_size=0.2, random_state=42)

train_X_final = np.array([np.vstack(i) for i in train_X.features_org_vel])
test_X_final = np.array([np.vstack(i) for i in test_X.features_org_vel])

X_train_org, X_test_org, y_train_org, y_test_org = train_X_final, test_X_final, train_y, test_y


#### split the data with calculated velocity and original velocity seperately 
# X_cal = data_final.features_cal_vel
# X_cal = np.array([np.vstack(i) for i in X_cal])

#X_org = data_final.features_org_vel
#X_org = np.array([np.vstack(i) for i in X_org])

#y = np.array(data_final.label)
# y = to_categorical(y)

# X_train_cal, X_test_cal, y_train_cal, y_test_cal = train_test_split(X_cal, y, test_size=0.2, random_state=42)

#X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(X_org, y, test_size=0.2, random_state=42)


##### make data into sequence for training
# X_train_cal = sequence.pad_sequences(X_train_cal, maxlen=50, padding='post', dtype='float', truncating='post')
# y_train_cal = np.array(y_train_cal).reshape(len(y_train_cal),1)

# X_test_cal = sequence.pad_sequences(X_test_cal, maxlen=50, padding='post', dtype='float', truncating='post')
# y_test_cal = np.array(y_test_cal).reshape(len(y_test_cal),1)



X_train_org = sequence.pad_sequences(X_train_org, maxlen=50, padding='post', dtype='float', truncating='post')
y_train_org = np.array(y_train_org).reshape(len(y_train_org),1)

X_test_org = sequence.pad_sequences(X_test_org, maxlen=50, padding='post', dtype='float', truncating='post')
y_test_org = np.array(y_test_org).reshape(len(y_test_org),1)


#### onehotecoder
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
enc = enc.fit(y_train_org)
y_train_org = enc.transform(y_train_org)
y_test_org = enc.transform(y_test_org)


# print('..........', y_train_cal.shape)

# enc_org = enc.fit(y_train_org)
# y_train_org = enc.transform(y_train_org)
# y_test_org = enc.transform(y_test_org)


### train model
import keras
model = keras.Sequential()
model.add(
      keras.layers.LSTM(
          units=128,
          input_shape=[X_train_org.shape[1], X_train_org.shape[2]]
      )
    )

model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dense(y_train_org.shape[1], activation='softmax'))
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['categorical_accuracy']
)

class_weights = [{ 
    0:1,
    1:1
},
{ 
    0:1,
    1:10
}]

for i in range(len(class_weights)):
    print('------------------------', i)
    history = model.fit(
        X_train_org, y_train_org,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        shuffle=False,
        class_weight = class_weights[i]
    )

    model.evaluate(X_test_org, y_test_org)
    y_pred_org = model.predict(X_test_org)


    predictions_org = y_pred_org[:,0]
    predictions_org[predictions_org>=0.5] = 1
    predictions_org[predictions_org<0.5] = 0

    testing_org = y_test_org[:,0]

    ### get the validation results to do feature engineering
    ### here validation results means False Positive results, we want to see if there are any patterns for those samples
    validation_final = test_X
    validation_final['prediction'] = predictions_org
    validation_final['ground_truth'] = testing_org
    validation_final = validation_final[(validation_final.prediction != validation_final.ground_truth) & (validation_final.ground_truth==1.0)].iloc[:,2:]
    validation_final.to_csv('results/results_validation/'+str(i)+'_validation_500ms_500ms.csv')  

 ### confusion matrix
    #cf_array_org = confusion_matrix(testing_org, predictions_org)
    #df_resutls = pd.DataFrame(cf_array_org)
    #df_resutls['accuracy'] = (df_resutls.iloc[0,0] + df_resutls.iloc[1,1])/np.sum(cf_array_org)
    #df_resutls['recall'] = df_resutls.iloc[0,0]/(df_resutls.iloc[0,0] + df_resutls.iloc[0,1])
    #df_resutls['precision'] = df_resutls.iloc[0,0]/(df_resutls.iloc[0,0] + df_resutls.iloc[1,0])
    #df_resutls.to_csv('results/results_1000/'+ str(i)+'_500ms_1000ms_original_vel'+'.csv')

    # print(predictions_cal.shape, testing_cal.shape)

