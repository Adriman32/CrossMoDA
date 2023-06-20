import os.path
import time

import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.layers import (Concatenate, Conv2D, Input, MaxPooling2D, UpSampling2D, Cropping2D)
from keras.models import Model
from tensorflow import keras





class UNet():
    def __init__(self, verbose = 0) -> None:
        self.verbose = verbose
        self.history = None
        self.model = self.create_model()
        self.cb = self.setup_cbs()

    def create_model(self):

        input_layer = Input(shape=(572,572,3))
        conv_layer_1 = Conv2D(filters=64,kernel_size=(3,3),padding='valid',activation='relu')(input_layer)
        conv_layer_2 = Conv2D(filters=64,kernel_size=(3,3),padding='valid',activation='relu')(conv_layer_1)

        pool_layer_1 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv_layer_2)
        conv_layer_3 = Conv2D(filters=128,kernel_size=(3,3),padding='valid',activation='relu')(pool_layer_1)
        conv_layer_4 = Conv2D(filters=128,kernel_size=(3,3),padding='valid',activation='relu')(conv_layer_3)

        pool_layer_2 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv_layer_4)
        conv_layer_5 = Conv2D(filters=256,kernel_size=(3,3),padding='valid',activation='relu')(pool_layer_2)
        conv_layer_6 = Conv2D(filters=256,kernel_size=(3,3),padding='valid',activation='relu')(conv_layer_5)

        pool_layer_3 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv_layer_6)
        conv_layer_7 = Conv2D(filters=512,kernel_size=(3,3),padding='valid',activation='relu')(pool_layer_3)
        conv_layer_8 = Conv2D(filters=512,kernel_size=(3,3),padding='valid',activation='relu')(conv_layer_7)

        pool_layer_4 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv_layer_8)
        conv_layer_9 = Conv2D(filters=1024,kernel_size=(3,3),padding='valid',activation='relu')(pool_layer_4)
        conv_layer_10 = Conv2D(filters=1024,kernel_size=(3,3),padding='valid',activation='relu')(conv_layer_9)

        upconv_layer_1 = UpSampling2D(size=(2,2))(conv_layer_10)
        crop_layer_1 = Cropping2D(cropping=((4,4),(4,4)))(conv_layer_8)
        concat_layer_1 = Concatenate()([crop_layer_1,upconv_layer_1])
        conv_layer_11 = Conv2D(filters=512,kernel_size=(3,3),padding='valid',activation='relu')(concat_layer_1)
        conv_layer_12 = Conv2D(filters=512,kernel_size=(3,3),padding='valid',activation='relu')(conv_layer_11)

        upconv_layer_2 = UpSampling2D(size=(2,2))(conv_layer_12)
        crop_layer_2 = Cropping2D(cropping=((16, 16), (16, 16)))(conv_layer_6)  
        concat_layer_2 = Concatenate()([crop_layer_2, upconv_layer_2])
        conv_layer_13 = Conv2D(filters=256, kernel_size=(3,3), padding='valid', activation='relu')(concat_layer_2)
        conv_layer_14 = Conv2D(filters=256, kernel_size=(3,3), padding='valid', activation='relu')(conv_layer_13)
            
        upconv_layer_3 = UpSampling2D(size=(2,2))(conv_layer_14)
        crop_layer_3 = Cropping2D(cropping=((40, 40), (40, 40)))(conv_layer_4)  
        concat_layer_3 = Concatenate()([crop_layer_3, upconv_layer_3])
        conv_layer_15 = Conv2D(filters=128, kernel_size=(3,3), padding='valid', activation='relu')(concat_layer_3)
        conv_layer_16 = Conv2D(filters=128, kernel_size=(3,3), padding='valid', activation='relu')(conv_layer_15)

        upconv_layer_4 = UpSampling2D(size=(2,2))(conv_layer_16)
        crop_layer_4 = Cropping2D(cropping=((88, 88), (88, 88)))(conv_layer_2)
        concat_layer_4 = Concatenate()([crop_layer_4, upconv_layer_4])
        conv_layer_17 = Conv2D(filters=64, kernel_size=(3,3), padding='valid', activation='relu')(concat_layer_4)
        conv_layer_18 = Conv2D(filters=64, kernel_size=(3,3), padding='valid', activation='relu')(conv_layer_17)

        conv_layer_19 = Conv2D(filters=2,kernel_size=(1,1),padding='valid',activation='relu')(conv_layer_18)

        model = Model(inputs=input_layer,outputs=conv_layer_19)

        return model


    def compile(self,optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'],verbose=False):
        if self.verbose or verbose:
            print("Compiling Model...")
        self.model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
        if self.verbose or verbose:
            print("Compilation Success!")
        return self.model
    
    def train (self,x_train,y_train,validation_data=None,epochs=10,batch_size=1,verbose = False):
        if self.verbose or verbose:
            print('Now Training Model...')
        self.history = self.model.fit(x_train, y_train, validation_data=validation_data,batch_size=batch_size, epochs=epochs, callbacks=[self.cb],verbose=verbose)
        if self.verbose or verbose:
            print('Training Complete!')
        return self.history
    
    def summary(self):
        model_summary = self.model.summary()
        return model_summary

    def save(self,model_name="",verbose=True):
        if(model_name == ""):
            model_name = 'output_models/U-Net_' + str(round(self.history.history['accuracy'][-1], 3)) + '.h5' 
        self.model.save(model_name)
        if self.verbose or verbose:
            print("Model saved to " + model_name)

    def setup_cbs(self):
        cb_list = []
        cb_list.append(self.setup_early_stopping_cb())
        cb_list.append(self.setup_tensorboard_cb())
        cb_list.append(self.setup_learning_rate_cb())
        return cb_list

    def setup_tensorboard_cb(self):
        model_name = 'U-Net_' + time.strftime("run_%Y_%m_%d_-%H_%M_%S")
        root_logdir = os.path.join(os.curdir, 'logs/')
        log_path = os.path.join(root_logdir, model_name)
        return TensorBoard(log_path)

    def setup_early_stopping_cb(self):
        monitor = 'val_loss'
        patience = 10
        best_weights = True
        return EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=best_weights)

    def setup_learning_rate_cb(self):
        monitor = 'val_loss'
        patience = 1
        factor = 0.5
        min_lr = 1e-4
        return ReduceLROnPlateau(monitor=monitor, patience=patience, factor=factor, min_lr=min_lr)