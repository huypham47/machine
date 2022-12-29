import numpy as np
import pandas as pd
from pathlib import Path
import os.path

from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf

from sklearn.metrics import r2_score
model = keras.models.load_model('D:/chuyen_nganh/20221/machine_learning/project/Nhap_mom_ML-hao/Nhap_mom_ML-hao/models2')
print('ok')
image_dir = Path('D:/chuyen_nganh/20221/machine_learning/project/Nhap_mom_ML-hao/Nhap_mom_ML-hao/age_prediction/test/002')
filepaths = pd.Series(list(image_dir.glob(r'*.jpg')), name='Filepath').astype(str)
ages = pd.Series(filepaths.apply(lambda x: os.path.split(os.path.split(x)[0])[1]), name='Age').astype(np.int)
images = pd.concat([filepaths, ages], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)

# image_dir = Path('D:/chuyen_nganh/20221/machine_learning/project/Nhap_mom_ML-hao/Nhap_mom_ML-hao/age_prediction/test/047')
# filepaths = pd.Series((image_dir), name='Filepath').astype(str)
# ages = pd.Series(name='Age')
# images = pd.concat([filepaths, ages], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)

train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

test_images = train_generator.flow_from_dataframe(
    dataframe=images,
    x_col='Filepath',
    y_col='Age',
    target_size=(120, 120),
    color_mode='rgb',
    class_mode='raw',
    batch_size=64,
    shuffle=False
)

predicted_ages = np.squeeze(model.predict(test_images))
true_ages = test_images.labels

# rmse = np.sqrt(model.evaluate(test_images, verbose=0))
# print("     Test RMSE: {:.5f}".format(rmse))

# r2 = r2_score(true_ages, predicted_ages)
# print("Test R^2 Score: {:.5f}".format(r2))
print(predicted_ages)
print(np.mean(predicted_ages))
# print(true_ages[0:100])
# print(model.evaluate(test_images, verbose=1))