import numpy as np
import pandas as pd
from os.path import join # exists , expanduser
#from tqdm import tqdm
#from sklearn.metrics import log_loss , accuracy_score
from keras.preprocessing import image
#from tensorflow.keras.applications import ResNet50
from os import listdir
import matplotlib.pyplot as plt
from keras.applications.vgg16 import  preprocess_input, decode_predictions
import cv2

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

def read_img(img_id, train_or_test, size):
    img =cv2.imread(join(data_dir, train_or_test, img_id + '.jpg'))
    img = cv2.resize(img,size)
   # img = image.img_to_array(img)

    return img
#read_img('000bec180eb18c7604dcecc8fe0dba07','train',(224,224))

#predictions


from sklearn.utils import shuffle
import tensorflow.keras.applications.inception_v3 as inception_v3
model = inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling='max', classes=1000)
true= []
def Run_model():
    i = 0
    for  img_id, breed,_,_ in shuffle(labels).tail(5).itertuples(index=False):
        i += 1
        img = read_img(img_id, 'train_set/train',(299,299))    
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


#using GPU
import tensorflow as tf
assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()

Run_model()
#Running the model

"""
#making application
from tkinter import *

# loading Python Imaging Library 
from PIL import ImageTk, Image 

# To get the dialog box to open when required 
from tkinter import filedialog 
def openfilename(): 

	# open file dialog box to select image 
	# The dialogue box has a title "Open" 
	filename = filedialog.askopenfilename() 
	return filename 
def open_img(): 
    # Select the Imagename  from a folder  
    x = openfilename() 
  
    # opens the image 
    img = Image.open(x) 
      
    # resize the image and apply a high-quality down sampling filter 
    img = img.resize((250, 250), Image.ANTIALIAS) 
  
    # PhotoImage class is used to add image to widgets, icons etc 
    img = ImageTk.PhotoImage(img) 
   
    # create a label 
    panel = Label(root, image = img) 
      
    # set the image as img  
    panel.image = img 
    panel.grid(row = 2) 
    
    def read_img(img):
        img =cv2.imread(img)
        img = cv2.resize(img,size)
        img = image.img_to_array(img)

    return img
    
    from sklearn.utils import shuffle
    import tensorflow.keras.applications.inception_v3 as inception_v3
    model = inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling='max', classes=1000)
    true= []
    def Run_model():
        i = 0
        for  img_id, breed,_,_ in shuffle(labels).itertuples(index=False):
            i += 1
            img = read_img(img_id, 'train_set/train',(299,299))    
            x = preprocess_input(np.expand_dims(img.copy(), axis=0))
            x = x/255
            preds = model.predict(x)
            _, imagenet_class_name, prob = decode_predictions(preds, top=0)[0][0]
            #cv2.putText(img,imagenet_class_name,(10,10),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255, 0, 0),thickness=1,bottomLeftOrigin = False)
            cv2.putText(img, breed,(10,10),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,0,0),thickness=1,bottomLeftOrigin = False)
            #cv2.imshow(imagenet_class_name,img)
            true.append(imagenet_class_name.lower() == breed)
            print(i, '  predicted_breed : ' + imagenet_class_name ,"||", 'original_breed : ' + breed)
           
        plt.show()


    
    Label(root, text = 'dummy' ).place(x = 400,y = 250)  
    
# Create a windoe 
root = Tk() 

# Set Title as Image Loader 
root.title("Image Loader") 
#text...........................

# Set the resolution of window 
root.geometry("600x600") 

# Allow Window to be resizable 
root.resizable(width = True, height = True) 

# Create a button and place it into the window using grid layout 
btn = Button(root, text ='open image', command = open_img).grid(row = 1, columnspan = 4) 
root.mainloop() 
"""