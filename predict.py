# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 16:55:41 2021

@author: User
"""
from os import listdir 
from os.path import isfile, join 
import numpy as np
import cv2 
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import keras

from tkinter import *
from keras.preprocessing import image
# loading Python Imaging Library 
from PIL import ImageTk, Image   
# To get the dialog box to open when required  
from tkinter import filedialog

modelPath = os.path.join("..", "model")
mypath= os.path.join("..","input/test_images")
inputPath=os.path.join("../input")
outPath = os.path.join("..", "output")

class_names = {0:'chris_evans',1:'chris_hemsworth',2:'homas_stanley_tom_holland',
               3:'jeremy_lee_renner',4:'mark_ruffalo',5:'robert_downey_jr',
               6:'scarlett_johansson'}


modelName = outPath+"/weights_6conv_20210617.hdf5"
model = load_model(modelName)
print('Model Loaded Sucessfully')

count=0

def open_img(): 
    # Select the Imagename  from a folder  
    x = openfilename() 
    # opens the image 
    img = Image.open(x) 
    im1 = img.save("test.jpg")
    img = ImageTk.PhotoImage(img) 
    # create a label 
    panel = Label(root, image = img)   
    # set the image as img  
    panel.image = img 
    panel.place(bordermode=OUTSIDE, x=80, y=100)
       
def openfilename(): 
  
    # open file dialog box to select image 
    # The dialogue box has a title "Open" 
    filename = filedialog.askopenfilename(title ='Select Image') 
    return filename
def prediction():
    img ="test.jpg"  
    test_images=[]
    test_labels=[]
    
    test_image = cv2.imread(img,cv2.IMREAD_COLOR)
    test_image = cv2.cvtColor(test_image,cv2.COLOR_BGR2RGB)
    test_image = cv2.resize(test_image, (224, 224))
    test_images.append(test_image)
    test_images=np.array(test_images)
    
    test_images=test_images.reshape((1,224,224,3))
    print(test_images[0].shape)
    test_images=test_images.astype('float32') / 255. #將色彩轉為0-1
    
    
    test_labels = ['0']
    test_labels = keras.utils.to_categorical(test_labels,7)
    test_labels=np.array(test_labels)
    # test_images=np.array(test_images)  
    print(test_images.shape)
    print(test_labels.shape)
    

    testLoss, testAcc = model.evaluate(test_images,test_labels,verbose=1)
    print("testLoss", testLoss)
    print("testAcc", testAcc)
    
    ##### Make prediction
    pc = model.predict_classes(test_images) #index of the label with max
    ps = model.predict(test_images) # the content of nodes in output layers
    print("Class of prediction: ", pc[0:7])
    print("Result of prediction: ", ps)
    print("Label of testing: ", test_labels)
    result = str(class_names[pc[0]])
    prediction = '預測成功'

    result = Label(root, text = prediction+"\n"+result, 
                   bg = '#FFF5EE',         #  背景顏色
                 font = ('Arial', 9),   # 字型與大小
                 width = 20, height = 2)   
    # set the image as img  
    result.place(bordermode=OUTSIDE, x=350, y=120)
# Create a window 
root = Tk()   
# Set Title as Image Loader 
root.title("復仇者聯盟人臉辨識")   
root.configure(bg='#B0E0E6')
# Set the resolution of window 
root.geometry("600x450")   
# Do't Allow Window to be resizable 
root.resizable(width = False, height = False) 
hello = Label(root,
              text = '歡迎使用，請點選上傳圖片按鈕，並上傳一張圖片' ,
                 bg = '#4682B4',         #  背景顏色
                 font = ('Arial', 9),   # 字型與大小
                 width = 70, height = 2)
# Create a button and place it into the window using place layout 
btn_open_image = Button(root, text ='上傳圖片', command = open_img).place( 
                                        x = 100, y= 400) 
btn_predict = Button(root, text ='Predict', command = prediction).place( 
                                        x = 200, y= 400) 
result_hd = Label(root, text = "結果")
hello.place(x=20, y=20)
result_hd.place(bordermode=OUTSIDE, x=400, y=90)
root.mainloop()