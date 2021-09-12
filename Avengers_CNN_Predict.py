# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 21:41:01 2021

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 31 18:11:09 2021

@author: reta
"""
#Import
#File
import os
inputPath=os.path.join("..","input")
outputPath=os.path.join("..","output")
#查找符合特定規則的文件路徑名
import glob

#Import 
#General Package
import numpy as np
import time
import matplotlib.pyplot as plt

#Import 
#Image Package
import cv2

#Import 
#Data Process
import h5py#儲存資料集
from random import shuffle#將序列的所有元素随機排序

#Import 
#ML Package
from collections import Counter
import math
import itertools#笛卡爾

import sklearn
from sklearn.model_selection import train_test_split#劃分數據集
from sklearn.metrics import classification_report#分類報告
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

import keras
from keras.preprocessing.image import ImageDataGenerator#圖片生成器
from keras.callbacks import LearningRateScheduler, ModelCheckpoint,TensorBoard,EarlyStopping
#回調函數返回學習速率；在每個訓練期之後保存模型

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam,RMSprop
from keras.utils import np_utils#可視化

#CNN+LSTM
from keras.layers import TimeDistributed
from keras.layers import LSTM

#VGG16
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

map_characters={0:'chris_evans',1:'chris_hemsworth',2:'jeremy_lee_renner',3:'mark_ruffalo',4:'robert_downey_jr',5:'scarlett_johansson',6:'homas_stanley_tom_holland'}

pic_size = 224#設定圖片大小
batch_size = 128 #batch_szie以2的次方做設定
#這裡 batch_size=128 ，表示我們要把128張隨機選擇的image放到一個batch裡面，
#然後把所有的image分成一個個不同的batch，Keras 會自動幫你完成隨機選擇image的過程
epochs = 200

num_classes = len(map_characters)
pictures_per_class = 300
test_size = 0.3 #資料切割7 3 分

def load_pictures(BGR):
    pics = []
    labels = []
    for k, char in map_characters.items():
        pictures = [k for k in glob.glob(inputPath+'/characters/%s/*' % char)]#從每類人物的資料夾裡返回所有圖片名字pictures=[****]
        #print(pictures)        
        #從pictures中選樣本集，如果樣本數目<pictures數目，則返回樣本數目；如果大於，則返回pictures數目
        nb_pic = round(pictures_per_class/(1-test_size)) if round(pictures_per_class/(1-test_size))<len(pictures) else len(pictures)
        # nb_pic = len(pictures)
        for pic in np.random.choice(pictures, nb_pic):#從每類pictures中随機選np_pic張圖片作為樣本數據集
            #以 cv2.imread 讀進來的資料，會存成NumPy陣列
             #讀取圖片，默認彩色圖,a.shape(x,x,3)
            a = cv2.imread(pic)
            if BGR:
                a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)#色彩空間轉換BGR轉為RGB
            a = cv2.resize(a, (pic_size,pic_size))#按比例縮放為pic_size * pic_size大小，此時a.shape(64,64,3)
            pics.append(a) #放入[]
            labels.append(k)
    return np.array(pics), np.array(labels) 

def get_dataset(save=True, load=False, BGR=True):
    if load:#讀取存好的資料集
        h5f = h5py.File(outputPath+'/dataset.h5','r')
        X_train = h5f['X_train'][:]
        X_test = h5f['X_test'][:]
        h5f.close()    

        h5f = h5py.File(outputPath+'/labels.h5','r')
        y_train = h5f['y_train'][:]
        y_test = h5f['y_test'][:]
        h5f.close()
        
    else:
        X, y = load_pictures(BGR)#讀取並獲得圖片信息
        '''
        print(X.shape,y.shape)
        '''
        y = keras.utils.to_categorical(y, num_classes)#轉换為one-hot編碼
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)#拆分數據集
        if save:
            h5f = h5py.File(outputPath+'/dataset.h5', 'w')
            h5f.create_dataset('X_train', data=X_train)
            h5f.create_dataset('X_test', data=X_test)
            h5f.close()

            h5f = h5py.File(outputPath+'/labels.h5', 'w')
            h5f.create_dataset('y_train', data=y_train)
            h5f.create_dataset('y_test', data=y_test)
            h5f.close()
            
    X_train = X_train.astype('float32') / 255.  #為了把pixel值轉成0-1
    X_test = X_test.astype('float32') / 255.
    print("Train", X_train.shape, y_train.shape)
    print("Test", X_test.shape, y_test.shape)
#    把每類的訓練集和測試集數目印出来
    if not load:
        dist = {k:tuple(d[k] for d in [dict(Counter(np.where(y_train==1)[1])), dict(Counter(np.where(y_test==1)[1]))]) 
                for k in range(num_classes)}
        print('\n'.join(["%s : %d train pictures & %d test pictures" % (map_characters[k], v[0], v[1]) 
            for k,v in sorted(dist.items(), key=lambda x:x[1][0], reverse=True)]))
    return X_train, X_test, y_train, y_test

def create_model_conv(input_shape):
    #input 層
    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding='same',activation='relu', input_shape=input_shape)) 
    
    #conv1
    model.add(Conv2D(32, (3, 3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    #conv2
    #Dropout->降低overfitting，配置為 0.5，表示每個神經元有 50% 的機率不參與下一層的傳遞。
    #這種技術迫使神經網絡需要學習更為穩健的特徵，因此可有效降低 Overfitting。
    model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
    
    # #conv3
    model.add(Conv2D(64, (3, 3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    # model.add(Dropout(0.5))

    #conv4
    model.add(Conv2D(256, (3, 3), padding='same',activation='relu')) 
    model.add(Conv2D(256, (3, 3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    
    #conv5
    model.add(Conv2D(512, (3, 3), padding='same',activation='relu')) 
    model.add(Conv2D(512, (3, 3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization()) #加快收歛速度，提高效率
    model.add(Dropout(0.5))
    
    #扁平層，將值拉成直線，做運算
    model.add(Flatten())
    #Dense 表示加一個Fully connected的layer
    model.add(Dense(1024))
    model.add(Activation('relu'))

    '''
    #model.add(Dense(256))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    
    model.add(Dense(128))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    '''
    model.add(Dense(num_classes, activation='softmax'))
    #opt = RMSprop(lr=0.0001, decay=1e-6)
    opt = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)#nesterov=True使用動量
    model.summary()
    return model, opt

def Conv_LSTM_model(input_shape=(None,10,224,244,3)):
    
    model = Sequential()
    print(input_shape)
    model.add(TimeDistributed(Conv2D(1, (2,2), activation='relu',input_shape=input_shape)))
   
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    
    model.add(TimeDistributed(Conv2D(1, (2,2),activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Dropout(0.25)))
    model.add(TimeDistributed(Flatten()))
   
    model.add(LSTM(50))
    model.add(Dense(num_classes,activation='sigmoid'))
    model.add(Dropout(0.2))
    
    opt = keras.optimizers.Adam(learning_rate=0.01)
    
    return model,opt

def create_model_vgg16(input_shape):
    conv_base=VGG16(weights='imagenet',input_shape=input_shape,include_top=False)
    model = Sequential()
    model.add(conv_base)
    '''
    opt = RMSprop(lr=0.0001, decay=1e-6)
    優化器用預設
    opt = SGD(lr=0.007, decay=1e-6, momentum=0.9, nesterov=True)#nesterov=True使用動量
    '''
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(num_classes, activation='softmax'))
    opt = Adam(lr=0.001)
    model.summary()
    return model, opt

def load_model_from_checkpoint(weights_path, six_conv=False, input_shape=(pic_size,pic_size,3)):
    if six_conv:
        model, opt = create_model_conv(input_shape)
    model.load_weights(weights_path)
    #optimizer 是選擇優化梯度下降的演算法，使用 adam
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    ##### Plot output
    saveFileName = outputPath + "/predict.jpg"
    plt.figure(figsize=(48,48))
    plt.title("Avengers Image Test")
    plt.figure(figsize=(10,10))
    for c in range(25):
        plt.subplot(5,5,c+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_test[c])
        # The CIFAR labels happen to be arrays, 
        # which is why you need the extra index
        a=np.where(y_test[c]==1)
        b=a[0][0]
        plt.xlabel(map_characters[b])
    plt.show()
    
#設置學習率衰减
def lr_schedule(epoch):
    initial_lrate = 0.08#初始學習率
    drop = 0.5#衰减為原來的多少倍
    epochs_drop = 12.0#每隔多久改變學習率
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))#math.pow(x,y)=x的y次方，math.floor向下取整
    #return lrate if lrate >= 0.0001 else 0.0001
    return lrate

def training(model, X_train, X_test, y_train, y_test, data_augmentation=False):
    #data augmentation來減少 overfitting
    if data_augmentation:
        #資料擴增法:
            #將現有資料圖片，藉由水平旋轉、上下翻轉之類的，把輸入很少的照片，增加成很多照片。
            #data augmantation 對資料量少很適合
        datagen = ImageDataGenerator(
            featurewise_center=False,  # 將輸入數據的均值設置為 0，依特徵進行
            samplewise_center=False,  # 將每個樣本的均值設置為 0
            featurewise_std_normalization=False,  # 將輸入除以數據標準差，依特徵進行
            samplewise_std_normalization=False,  # 將每個輸入除以其標準差
            # zca_whitening=False,  #應用 ZCA 白化
            rotation_range=12,  # 随機旋轉的度數範圍(degrees, 0 to 180)，旋轉角度
            width_shift_range=0.13,  # 随機水平移動的範圍，比例
            height_shift_range=0.13, # 随機垂直移動的範圍，比例
            zoom_range=0.2,
            horizontal_flip=True,  # 随機水平翻轉，相當於镜像
            vertical_flip=False)  # 随機垂直翻轉，相當於镜像
      
        datagen.fit(X_train)
        
        ###每當val_cc有提升就保存checkpoint
        #save_best_only=True被監測數據的最佳模型就不會被覆蓋，mode='max'保存的是準確率最大值
        filepath=outputPath+"/weights_6conv_%s.hdf5" % time.strftime("%Y%m%d") 
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')    
        '''
        # 產生日誌藉助 TensorBoard 進行可視化分析。
        tensorboard = TensorBoard(log_dir='logs', histogram_freq=0)
        #自動調整學習率
         lrate = LearningRateScheduler(lr_schedule,verbose=1)
         
        #EarlyStopping
        #當val_loss不再下降，提前停止訓練
        #超過20次就停止訓練
       
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')
        
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, verbose=1, mode='max')
        '''
        callbacks_list = [checkpoint]
        history = model.fit_generator(datagen.flow(X_train, y_train,#傳入 Numpy 數據和標籤數組，生成批次的 增益的/標準化的數據。在生成的批次數據上無限制地無限次循环。
                                    batch_size=batch_size), 
                                    steps_per_epoch=X_train.shape[0] // batch_size,
                                    epochs=epochs,
                                    validation_data=(X_test, y_test),
                                    verbose=1,#輸出進度條
                                    callbacks=checkpoint)#調用一些列回調函数
        #callback會有一些回傳值，根據回傳值調整每次更新的頻率
        #查看分類報告，返回每類的精确率，召回率，F1值
        #P=TP/(TP+FP),R=TP/(TP+FN),F1=2PR/(P+R)
        score = model.evaluate(X_test,y_test,verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        y_pred = model.predict(X_test)
        print('\n', sklearn.metrics.classification_report(np.where(y_test > 0)[1], #y_test真實分類，np.where返
                                                          #回（array[],array[]），其中後面的array就是行方向上，y_test>0(1)的索引
                                                          np.argmax(y_pred, axis=1),#返回行方向上最大數值的索引
                                                          target_names=list(map_characters.values())), sep='') 
    
        #acc和loss可視化
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # plt.savefig(('acc_%s.png') % time.strftime("%Y%m%d") )
        
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.legend(['train', 'test'], loc='upper left')
        plt.ylim([0.1, 5])
        plt.show()
        # plt.savefig(('loss_%s.png') % time.strftime("%Y%m%d") )
        
        #畫出混淆矩陣
        plt.figure(figsize = (10,10))
        cnf_matrix = sklearn.metrics.confusion_matrix(np.where(y_test > 0)[1],np.argmax(y_pred, axis=1))
        classes = list(map_characters.values())
        thresh = cnf_matrix.max() / 2.#擴值
        for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
            plt.text(j, i, cnf_matrix[i, j],#在圖形中加註釋
                     horizontalalignment="center",#水平對齊
                     color="white" if cnf_matrix[i, j] > thresh else "black")
        plt.imshow(cnf_matrix,interpolation='nearest',cmap=plt.cm.Blues)#cmap顏色圖譜，默認RGB(A)
        plt.colorbar()#顯示顏色條
        plt.title('confusion_matrix')#標題
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks,classes,rotation=90)
        plt.yticks(tick_marks,classes)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(('confusion_matrix_%s.png') % time.strftime("%Y%m%d") )
        
        '''
        cnf_matrix = sklearn.metrics.confusion_matrix(np.where(y_test > 0)[1],y_true=np.argmax(y_pred, axis=1))
        classes = list(map_characters.values())
        plot_confusion_matrix(cnf_matrix,classes)
        '''
        for c in range(25):
            plt.subplot(5,5,c+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(X_test[c])
            # The CIFAR labels happen to be arrays, 
            # which is why you need the extra index
            a=np.where(y_test[c]==1)
            b=a[0][0]
            plt.xlabel(map_characters[b])
        plt.show()
        
    else:
        history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1,#用作驗證集的訓練數據的比例
          verbose=1,
          shuffle=True)#是否在每輪進行數據混洗
        score = model.evaluate(X_test, y_test, verbose=1)
        
        #查看分類報告，返回每類的精確率，召回率，F1值
        #P=TP/(TP+FP),R=TP/(TP+FN),F1=2PR/(P+R)
        score = model.evaluate(X_test,y_test,verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        y_pred = model.predict(X_test)
        print('\n', sklearn.metrics.classification_report(np.where(y_test > 0)[1], #y_test真實分類，np.where返
                                                          #回（array[],array[]），其中後面的array就是行方向上，y_test>0(1)的索引
                                                          np.argmax(y_pred, axis=1),#返回行方向上最大數值的索引
                                                          target_names=list(map_characters.values())), sep='') 

        model.save_weights(outputPath+"CNN_model.h5")
        print("Saved model to disk")
        
        #acc和loss可視化
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.savefig(('acc_%s.png') % time.strftime("%Y%m%d") )

        # summarize history for loss
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.legend(['train', 'test'], loc='upper left')
        plt.ylim([0.1, 5])
        plt.show()
        plt.savefig(('loss_%s.png') % time.strftime("%Y%m%d") )
        
        #畫出混淆矩陣
        plt.figure(figsize = (10,10))
        cnf_matrix = sklearn.metrics.confusion_matrix(np.where(y_test > 0)[1],np.argmax(y_pred, axis=1))
        classes = list(map_characters.values())
        thresh = cnf_matrix.max() / 2.#擴增值
        for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
            plt.text(j, i, cnf_matrix[i, j],#在圖形中添加註解
                     horizontalalignment="center",#水平對齊
                     color="white" if cnf_matrix[i, j] > thresh else "black")
        plt.imshow(cnf_matrix,interpolation='nearest',cmap=plt.cm.Blues)#cmap颜色圖譜，默認RGB(A)
        plt.colorbar()#顯示顏色條
        plt.title('confusion_matrix')#標題
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks,classes,rotation=90)
        plt.yticks(tick_marks,classes)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        #y_test predict
        for c in range(25):
            plt.subplot(5,5,c+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(X_test[c])
            # The CIFAR labels happen to be arrays, 
            # which is why you need the extra index
            a=np.where(y_test[c]==1)
            b=a[0][0]
            plt.xlabel(map_characters[b])
        plt.show()
    return model, history

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_dataset(load=True)
    #CNN(Conv2d)
    # model, opt = create_model_conv(X_train.shape[1:])
    
    #VGG16 model
    model, opt = create_model_vgg16(X_train.shape[1:])
    # model,opt = load_model_from_checkpoint('../output/weights_6conv_20210615.hdf5', six_conv=True, input_shape=(pic_size,pic_size,3))
    
    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
    #LSTM+CNN
    '''
    model,opt=Conv_LSTM_model((1,224,224,3))

    model.compile(loss='mean_squared_error',
              optimizer=opt,
              metrics=['accuracy'])
    X_train=X_train.reshape(1465,1,224,224,3)
    X_test=X_test.reshape(628,1,224,224,3)
    
    '''
    model, history = training(model, X_train, X_test, y_train, y_test, data_augmentation=True)
    

