import numpy as np
import pandas as pd
from pathlib import Path
import os.path
from tensorflow import keras
from sklearn.model_selection import train_test_split

import tensorflow as tf

from sklearn.metrics import r2_score
model = keras.models.load_model('D:/chuyen_nganh/20221/machine_learning/project/Nhap_mom_ML-hao/Nhap_mom_ML-hao/models')

image_dir = Path('D:/chuyen_nganh/20221/machine_learning/project/Nhap_mom_ML-hao/Nhap_mom_ML-hao/age_prediction/train/11-15')
filepaths = pd.Series(list(image_dir.glob(r'**/*.jpg')), name='Filepath').astype(str)
ages = pd.Series(filepaths.apply(lambda x: os.path.split(os.path.split(x)[0])[1]), name='Age').astype(np.int)
images = pd.concat([filepaths, ages], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)
print(images)
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_images = train_generator.flow_from_dataframe(
    dataframe=images,
    x_col='Filepath',
    y_col='Age',
    target_size=(120, 120),
    color_mode='rgb',
    class_mode='raw',
    batch_size=64,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=images,
    x_col='Filepath',
    y_col='Age',
    target_size=(120, 120),
    color_mode='rgb',
    class_mode='raw',
    batch_size=64,
    shuffle=True,
    seed=42,
    subset='validation'
)

print(len(train_images))


print('okkkk')
history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=100
)

model.save('models')