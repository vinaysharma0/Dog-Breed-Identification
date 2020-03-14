import numpy as np
import pandas as pd
from os.path import join

from keras.preprocessing import image
import keras
#from tensorflow.keras.applications import ResNet50
from os import listdir
from keras.applications.vgg16 import  preprocess_input, decode_predictions

import cv2

#.......................Remove all the comments and comment all the code which is not commented except the modules if you want to train your own model and then test it on the dataset...................#


"""
#....................................................
input_size = 224
num_of_class = 15
data_dir = 'data/'
labels = pd.read_csv('labels.csv')
sample_sub = pd.read_csv('sample_submission.csv')
#checking length of images and csv's
print(len(listdir('data/train_set')),len(labels))
print(len(listdir('data/test_set')),len(sample_sub))

#....................................................

selected_breed_list = list(labels.groupby('breed').count().sort_values(by='id', ascending=False).index)
labels = labels[labels['breed'].isin(selected_breed_list)]
labels['target'] = 1
labels['rank'] = labels.groupby('breed').rank()
labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0)
"""
"""
def read_img(img_id, train_or_test, size):
    img =cv2.imread(join(data_dir, train_or_test, img_id + '.jpg'))
    img = cv2.resize(img,size)
    return img
"""

#predictions

from sklearn.utils import shuffle
from tensorflow.keras.applications import NASNetLarge
model = NASNetLarge(weights='imagenet')

# Lower loop can be used if you want to test the data on your Algorithm.
"""
true= []
def Run_model(num_of_img):
    i = 0
    for  img_id, breed,_,_ in shuffle(labels).head(num_of_img).itertuples(index=False):
        i += 1
        img = read_img(img_id, 'train_set/train',(331,331))    
        x = preprocess_input(np.expand_dims(img.copy(), axis=0))
        x = x / 255
        preds = model.predict(x)
        _, imagenet_class_name, prob = decode_predictions(preds, top=0)[0][0]
        cv2.putText(img,imagenet_class_name,(10,10),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255, 0, 0),thickness=1,bottomLeftOrigin = False)
        cv2.putText(img, breed,(10,10),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,0,0),thickness=1,bottomLeftOrigin = False)
        cv2.imshow(imagenet_class_name,img)
        true.append(imagenet_class_name.lower() == breed)
        print(i, '  predicted_breed : ' + imagenet_class_name ,"||", 'original_breed : ' + breed)
       
    plt.show()
"""
import tensorflow as tf
assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()

config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} ) 
sess = tf.compat.v1.Session(config=config) 
keras.backend.set_session(sess)


# importing only those functions 
from tkinter import *
  
# loading Python Imaging Library 
from PIL import ImageTk, Image 
  
# To get the dialog box to open when required  
from tkinter import filedialog 

def openfilename(): 

	# open file dialog box to select image 
	# The dialogue box has a title "Open" 
	filename = filedialog.askopenfilename(title ='Open') 
	return filename 
def open_img():
    x = openfilename()
    img = Image.open(x)
    img = img.resize((331,331), Image.ANTIALIAS)
    y = preprocess_input(np.expand_dims(img, axis=0))
    y = y / 225
    pred = model.predict(y)
    _, pred_name, prob = decode_predictions(pred,top=0)[0][0]
    print(pred_name)
    img = ImageTk.PhotoImage(img)
    #img = image.load_img(img)
    
    
    #img = read_img(img)
    panel = Label(root,image = img)
    panel.image = img
    panel.grid(row = 3) 
    text = StringVar(root)
    text.set(pred_name)
    l = Label(root, textvariable=text, font=("comic sans ms", 10))
    l.place(x=400,y=90)
    
    


# Create a windoe 
root = Tk() 
  
# Set Title as Image Loader 
root.title("Image Loader") 
  
# Set the resolution of window 
root.geometry("550x300") 

# Allow Window to be resizable 
root.resizable(width = True, height = True) 
  
# Create a button and place it into the window using grid layout 
btn = Button(root, text ='open image', command = open_img).grid( 
                                        row = 1, columnspan = 4) 
root.mainloop() 
















