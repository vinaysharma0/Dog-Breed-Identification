import numpy as np
import pandas as pd
import datetime as dt
from mpl_toolkits.axes_grid1 import ImageGrid
from os.path import join, exists , expanduser
from tqdm import tqdm
from sklearn.metrics import log_loss , accuracy_score
from keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from sklearn.linear_model import LogisticRegression
from os import listdir, makedirs
import matplotlib.pyplot as plt
from keras.applications.vgg16 import  preprocess_input, decode_predictions
import cv2

#....................................................
input_size = 224
num_of_class = 16
SEED = 1987
data_dir = 'data/'
labels = pd.read_csv('labels.csv')
sample_sub = pd.read_csv('sample_submission.csv')
#checking length of images and csv's
print(len(listdir('data/train')),len(labels))
print(len(listdir('data/test')),len(sample_sub))

#....................................................

selected_breed_list = list(labels.groupby('breed').count().sort_values(by='id', ascending=False).head(num_of_class).index)
labels = labels[labels['breed'].isin(selected_breed_list)]
labels['target'] = 1
labels['rank'] = labels.groupby('breed').rank()
labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0)
np.random.seed(seed=SEED)
rnd = np.random.random(len(labels))
train_idx = rnd < 0.8
valid_idx = rnd >= 0.8
y_train = labels_pivot[selected_breed_list].values
ytr = y_train[train_idx]
yv = y_train[valid_idx]

def read_img(img_id, train_or_test, size):
    """Read and resize image.
    # Arguments
        img_id: string
        train_or_test: string 'train' or 'test'.
        size: resize the original image.
    # Returns
        Image as numpy array.
    """
    img = image.load_img(join(data_dir, train_or_test, img_id + '.jpg'), target_size=size)
   # img = image.img_to_array(img)
    return img
#read_img('000bec180eb18c7604dcecc8fe0dba07','train',(224,224))

#predictions

model = ResNet50(weights='imagenet')
j = int(np.sqrt(num_of_class))
i = int(np.ceil(1. * num_of_class / j))
fig = plt.figure(1, figsize=(16, 16))
grid = ImageGrid(fig, 111, nrows_ncols=(i, j), axes_pad=0.05)

for i, (img_id, breed) in enumerate(labels['id', 'breed'].values):
    ax = grid[i]
    img = read_img(img_id, 'train', (224, 224))
    
    ax.imshow(img)
    x = preprocess_input(np.expand_dims(img.copy(), axis=0))
    preds = model.predict(x)
    _, imagenet_class_name, prob = decode_predictions(preds, top=1)[0][0]
    ax.text(10, 180, 'ResNet50: %s (%.2f)' % (imagenet_class_name , prob), color='w', backgroundcolor='k', alpha=0.8)
    ax.text(10, 200, 'LABEL: %s' % breed, color='k', backgroundcolor='w', alpha=0.8)
    ax.axis('off')
plt.show()











