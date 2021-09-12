# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 15:18:10 2021

@author: reta
"""
from os import listdir 
from os.path import isfile, join 
import numpy as np
import cv2 
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import glob
import keras

modelPath = os.path.join("..", "model")
mypath= os.path.join("..","input/test_images")
inputPath=os.path.join("../input")
outPath = os.path.join("..", "output")

class_names = {0:'chris_evans',1:'chris_hemsworth',2:'homas_stanley_tom_holland',
               3:'jeremy_lee_renner',4:'mark_ruffalo',5:'robert_downey_jr',6:'scarlett_johansson'}

test_images=[]
test_labels=[]
pic_size=224

onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ] 
images = np.empty(len(onlyfiles), dtype=object) 

for n in range(0, len(onlyfiles)): 
    images[n] = cv2.imread(join(mypath,onlyfiles[n]),cv2.IMREAD_COLOR)
    images[n] = cv2.cvtColor(images[n],cv2.COLOR_BGR2RGB)
    images[n] = cv2.resize(images[n], (224, 224))
    test_images.append(images[n])
test_images =np.array(test_images)    

plt.imshow(test_images[0])

test_images = test_images.reshape((1,224,224,3))
test_images = test_images.astype('float32') / 255. #將色彩轉為0-1

test_labels = ['0']
test_labels = keras.utils.to_categorical(test_labels,7)
test_labels=np.array(test_labels)


print(test_images.shape)
print(test_labels.shape)

modelName = outPath+"/weights_6conv_20210617.hdf5"
trainedModel = load_model(modelName)
print("model summary", trainedModel.summary())

##### Evaluate trained model
"""測試資料及正確率比訓練資料正確率有落差，表示可能有overlifting的現象
"""

testLoss, testAcc = trainedModel.evaluate(test_images,test_labels,verbose=1)
print("testLoss", testLoss)
print("testAcc", testAcc)

##### Make prediction
pc = trainedModel.predict_classes(test_images) #index of the label with max
ps = trainedModel.predict(test_images) # the content of nodes in output layers
print("Class of prediction: ", pc[0:7])
print("Result of prediction: ", ps)
print("Label of testing: ", test_labels)


##### Plot output
saveFileName = outPath + "/image_0.jpg"
plt.figure(figsize=(27,10))
plt.title("Avengers Image Test")
    
ax = plt.subplot(1,1,1)
ax.imshow(test_images[0]) 
ax.set_title("label={}\n predi={}\n characters={}".format(str(test_labels[0]), str(pc[0]),
             str(class_names[pc[0]])), fontsize=18)
ax.set_xticks([])
ax.set_yticks([])    
    
plt.savefig("Result.jpg")
plt.show()