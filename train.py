import argparse
from typing import Tuple

import tensorflow as tf
import numpy as np
import pickle 
import math
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage.transform import resize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LearningRateScheduler
from scipy.ndimage.interpolation import zoom
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import Adam,SGD
#from keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D,Add,concatenate,MaxPooling2D,Activation,GlobalAveragePooling2D,LocallyConnected2D,SeparableConv2D
from tensorflow.keras.layers import Dropout,Cropping2D, Input,DepthwiseConv2D, AveragePooling2D, Flatten, Dense, Reshape, BatchNormalization,Multiply
from tensorflow.keras.layers import UpSampling2D,ZeroPadding2D,Dot,Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental import RandomFourierFeatures
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
import albumentations as albu
from dataset import build_dataset


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('tfrec_roots',
        help='path(s) to folder with tfrecords '
             '(can be one or multiple gcs bucket paths as well)',
        nargs='+')
    arg('--image-size', type=int, default=224)

    arg('--batch-size', type=int, default=512, help='per device')
    arg('--lr', type=float, default=1.6)
    arg('--lr-decay', type=float, default=0.9)
    arg('--epochs', type=int, default=110)
    arg('--lr-sustain-epochs', type=int, default=20)
    arg('--lr-warmup-epochs', type=int, default=5)

    # TODO move to dataset.json created during pre-processing
    arg('--n-classes', type=int, default=1000)
    arg('--n-train-samples', type=int, default=1281167)

    arg('--xla', type=int, default=0, help='enable XLA')
    arg('--mixed', type=int, default=1, help='enable mixed precision')
    args = parser.parse_args()

    strategy, tpu = get_strategy()
    # setup_policy(xla_accelerate=args.xla, mixed_precision=args.mixed, tpu=tpu)

    image_size = (args.image_size, args.image_size)
    batch_size = args.batch_size * strategy.num_replicas_in_sync
    dtype = tf.float32
    if args.mixed:
        dtype = tf.bfloat16 if tpu else tf.float16
    train_dataset, valid_dataset = [build_dataset(
        args.tfrec_roots,
        is_train=is_train,
        image_size=image_size,
        cache=not is_train,
        batch_size=batch_size,
        drop_filename=True,
        dtype=dtype,
        ) for is_train in [True, False]]

    model = build_model(
        strategy, image_size=image_size, n_classes=args.n_classes)
    lr_schedule = build_lr_schedule(
        lr_max=args.lr,
        lr_warmup_epochs=args.lr_warmup_epochs,
        lr_sustain_epochs=args.lr_sustain_epochs,
        lr_decay=args.lr_decay,
    )
    lr_callback = tf.keras.callbacks.LearningRateScheduler(
        lr_schedule, verbose=True)

    # TODO "imagenet" preprocessing
    # TODO L2 weight decay
    model.fit(
        train_dataset,
        steps_per_epoch=args.n_train_samples // batch_size,
        epochs=args.epochs,
        callbacks=[lr_callback],
        validation_freq=4,
        validation_data=valid_dataset,
    )





def bock_one(input_data,F1,use_bias, activation, drop_out):
    init ="he_uniform"

    conv1 = Conv2D(F1,(1,1),strides=(1, 1),padding="same",
                use_bias=use_bias,
                kernel_initializer=init)(input_data)

#     conv1 = BatchNormalization()(conv1)
#     conv1 = Activation(activation)(conv1)

    conv = Conv2D(F1,(1,1),strides=(1, 1),padding="same",
                use_bias=use_bias,
                kernel_initializer=init)(conv1)
    conv = BatchNormalization()(conv)

#     conv = DepthwiseConv2D((3,3),strides=(1, 1),padding="same",
#                            use_bias=use_bias,
#                            kernel_initializer=init)(conv)
    
    conv = Add()([conv,conv1])
    
    conv2 = Conv2D(F1,(1,1),strides=(1, 1),padding="same",
                use_bias=use_bias,
                kernel_initializer=init)(conv)
    
#     conv2 = BatchNormalization()(conv2)
#     conv2 = Activation(activation)(conv2)

    conv = Conv2D(F1,(1,1),strides=(1, 1),padding="same",
                use_bias=use_bias,
                kernel_initializer=init)(conv2)
    conv = BatchNormalization()(conv)
    
    conv = block_two(conv,use_bias, activation, drop_out)
    # for  _ in range(3):
    conv = Conv2D(F1,(1,1),strides=(1, 1),padding="same",
                use_bias=use_bias,
                kernel_initializer=init)(conv)

    conv = BatchNormalization()(conv)
#     conv = Activation(activation)(conv)
    # conv = BatchNormalization()(conv)
    conv = Conv2D(F1,(1,1),strides=(1, 1),padding="same",
                use_bias=use_bias,
                kernel_initializer=init)(conv)
    conv = BatchNormalization()(conv)
    conv = Add()([conv,conv2])
    conv = Activation(activation)(conv)

    conv4 = Conv2D(F1,(1,1),strides=(1, 1),padding="same",
                use_bias=use_bias,
                kernel_initializer=init)(conv)
    
#     conv4 = BatchNormalization()(conv4)
#     conv4 = Activation(activation)(conv4)


    conv = Conv2D(F1,(1,1),strides=(1, 1),padding="same",
                use_bias=use_bias,
                kernel_initializer=init)(conv4)
    conv = conv = BatchNormalization()(conv)
    
    
#     conv = DepthwiseConv2D((3,3),strides=(1, 1),padding="same",
#                            use_bias=use_bias,
#                            kernel_initializer=init)(conv)
                
    conv = Add()([conv,conv4])
#     conv = Conv2D(F1,(1,1),strides=(1, 1),
#                   padding="same",
#                   use_bias=use_bias,
#                   kernel_initializer="he_uniform")(conv)
    # for _ in range(3):
    conv = Conv2D(F1,(1,1),strides=(1, 1),
                padding="same",
                use_bias=use_bias,
                kernel_initializer=init)(conv)
    
#     conv = BatchNormalization()(conv)
#     conv = Activation(activation)(conv)

    conv = Conv2D(F1,(1,1),strides=(1, 1),padding="same",
                use_bias=use_bias,
                kernel_initializer=init)(conv)
                
#     conv = DepthwiseConv2D((3,3),strides=(1, 1),padding="same",
#                            use_bias=use_bias,
#                            kernel_initializer=init)(conv)
    
    # kernel_regularizer=l2(0.001)
    conv = BatchNormalization()(conv)
    conv = Activation(activation)(conv)
    conv = Dropout(.5)(conv)
    conv = Add()([input_data,conv])
    
    #dot  = Dot(axes=1)([input_data,conv])
    #reshape = Reshape((28,28,64))(dot)
    return conv




def block_two(input_data,use_bias, activation, drop_out):
    init ="he_uniform"

    conv = DepthwiseConv2D((5,5),strides=(1, 1),padding="same",
                        use_bias=use_bias,
                        kernel_initializer=init)(input_data)
    
    
    conv = BatchNormalization()(conv)
    conv = Activation(activation)(conv)
#     conv = Dropout(drop_out)(conv)
    F1 = conv.shape[3]
    conv = Conv2D(F1,(1,1),strides=(1, 1),padding="same",
                use_bias=use_bias,
                kernel_initializer=init)(conv)
    
    conv = DepthwiseConv2D((3,3),strides=(1, 1),padding="same",
                        use_bias=use_bias,
                        kernel_initializer=init)(conv)

    conv = BatchNormalization()(conv)
    conv = Activation(activation)(conv)
    conv = Dropout(.5)(conv)
    #conv = Add()([input_data,conv])
    
    return conv

def  dense_layer(input_data,x):
    xx= input_data.shape[1]
    
#     dense = Dense(x, activation = "relu")(input_data)
#     dense = Dense(2*x, activation = "relu")(dense)
#     dense = Dense(int(2.5*x), activation = "relu")(dense)
#     dense = Dense(3*x, activation = "relu")(dense)
#     dense = Dense(int(3.5*x), activation = "relu")(dense)
    reshape = Reshape((1,1,int(xx)))(input_data)
    return reshape

def conv_layer(input_data, F1):
    conv = Conv2D(F1,(1,1),strides=(1, 1),
                padding="same",
                use_bias=False,
                kernel_initializer ="he_uniform")(input_data)
    conv = BatchNormalization()(conv)
    
    return conv


def build_model(strategy):

    try: # detect TPUs
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect() # TPU detection
        strategy = tf.distribute.TPUStrategy(tpu)
    except ValueError: # detect GPUs
        strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines

    print("Number of accelerators: ", strategy.num_replicas_in_sync)
    act = "elu"
    init = "he_uniform"
    with strategy.scope():
        input_data =Input((224,224,3))

        conv_1 = conv_layer(input_data,256) #64x64
        conv_2 = bock_one(conv_1,256,False, act, .25)#64x64
        conv_3 = block_two(conv_2,False, act, .25)#64x64
        max_1  = AveragePooling2D()(conv_3)#64x64---->>#16x16

        ################################# 16x16
        conv_5 = bock_one(max_1,256,False, act, .25)#16x16
        conv_6 = conv_layer(conv_5, 256)#16x16

        add_1  = Add()([max_1,conv_6])#  32X32
        conv_7 = conv_layer(add_1, 48)
        conv_77 = bock_one(conv_7,48,False, act, .25)
        conv_77 = block_two(conv_77,False, act, .25)
        # max_2  = MaxPooling2D()(conv_7)
        max_2  = AveragePooling2D()(conv_77)
        #############################
        conv_8 = bock_one(max_2,48,False, act, .25)
        conv_8 = conv_layer(conv_8, 48)
        global_avg_1 = GlobalAveragePooling2D()(conv_8)

        dense_1 =dense_layer(global_avg_1,128)  #128

        #     global_avg_2 = GlobalAveragePooling2D()(dense_1)
        #     reshape_1 = Reshape((1,1,448))(global_avg_2)

        conv_9 = Conv2D(48,(1,1),strides=(1, 1),padding="same",
                        use_bias=False,
                        kernel_initializer=init)(dense_1)
        conv_10 = Conv2D(48,(1,1),strides=(1, 1),padding="same",
                        use_bias=False,
                        kernel_initializer=init)(conv_9)

        add_2= Add()([conv_77,conv_10])

        '''
        **************************************************************************
        '''


        conv_11 = conv_layer(add_2,64) #64x64
        conv_21 = bock_one(conv_11,64,False, act, .25)#64x64
        conv_31 = block_two(conv_21,False, act, .25)#64x64
        max_11  = AveragePooling2D()(conv_31)#64x64---->>#16x16

        conv_51 = bock_one(max_11,64,False, act, .25)#16x16
        conv_61 = conv_layer(conv_51, 64)#16x16

        add_11  = Add()([max_11,conv_61])#  32X32
        conv_71 = conv_layer(add_11, 80)
        conv_771 = bock_one(conv_71,80,False, act, .25)
        conv_771 = block_two(conv_771,False, act, .25)
        # max_2  = MaxPooling2D()(conv_7)
        max_21  = AveragePooling2D()(conv_771)

        conv_81 = bock_one(max_21,80,False, act, .25)
        conv_81 = conv_layer(conv_81, 80)
        global_avg_11 = GlobalAveragePooling2D()(conv_81)

        dense_11 =dense_layer(global_avg_11,128)  #128

        conv_91 = Conv2D(80,(1,1),strides=(1, 1),padding="same",
                        use_bias=False,
                        kernel_initializer=init)(dense_11)

        conv_101 = Conv2D(80,(1,1),strides=(1, 1),padding="same",
                        use_bias=False,
                        kernel_initializer=init)(conv_91)

        add_21= Add()([conv_771,conv_101])



        '''
        *************************************************************************
        '''



        conv_12 = conv_layer(add_21,192) #64x64
        conv_22 = bock_one(conv_12,192,False, act, .25)#64x64
        conv_32 = block_two(conv_22,False, act, .25)#64x64
        max_12  = AveragePooling2D()(conv_32)#64x64---->>#16x16

        conv_52 = bock_one(max_12,192,False, act, .25)#16x16
        conv_62 = conv_layer(conv_52, 192)#16x16

        add_12  = Add()([max_12,conv_62])#  32X32
        conv_72 = conv_layer(add_12, 224)
        conv_772 = bock_one(conv_72,224,False, act, .25)
        conv_772 = block_two(conv_772,False, act, .25)
        # max_2  = MaxPooling2D()(conv_7)
        max_22  = AveragePooling2D()(conv_772)

        conv_82 = bock_one(max_22,224,False, act, .25)
        conv_82 = conv_layer(conv_82, 224)
        global_avg_12 = GlobalAveragePooling2D()(conv_82)

        dense_12 =dense_layer(global_avg_12,224)  #128

        #     global_avg_2 = GlobalAveragePooling2D()(dense_1)
        #     reshape_1 = Reshape((1,1,448))(global_avg_2)

        conv_92 = Conv2D(224,(1,1),strides=(1, 1),padding="same",
                        use_bias=False,
                        kernel_initializer=init)(dense_12)
        conv_102 = Conv2D(224,(1,1),strides=(1, 1),padding="same",
                        use_bias=False,
                        kernel_initializer=init)(conv_92)

        add_22= Add()([conv_772,conv_102])



        '''
        *************************************************************************
        '''



        conv_13 = conv_layer(add_22,380) #64x64
        conv_23 = bock_one(conv_13,380,False, act, .5)#64x64
        conv_33 = block_two(conv_23,False, act, .5)#64x64
        max_13  = AveragePooling2D()(conv_33)#64x64---->>#16x16

        conv_53 = bock_one(max_13,380,False, act, .5)#16x16
        conv_63 = conv_layer(conv_53, 380)#16x16

        add_13  = Add()([max_13,conv_63])#  32X32
        conv_73 = conv_layer(add_13, 420)
        conv_773 = bock_one(conv_73,420,False, act, .5)
        conv_773 = block_two(conv_773,False, act, .5)
        # max_2  = MaxPooling2D()(conv_7)
        max_23  = AveragePooling2D()(conv_773)
        conv_83 = conv_layer(max_23, 480)
        conv_83 = bock_one(conv_83,480,False, act, .5)
        conv_83 = conv_layer(conv_83, 480)


        conv = GlobalAveragePooling2D()(conv_83)
        conv = Dropout(.75)(conv)
        # conv = Dense(1024, activation ="swish")(conv)
        dense = Dense(1000, activation ="softmax")(conv)
                                                        #categorical_crossentropy
        model = Model(inputs=input_data, outputs=dense)#sparse_categorical_crossentropy,,  Adam(lr=0.0001,decay=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    model.compile(optimizer=SGD(),#SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True),
                loss='sparse_categorical_crossentropy', metrics=["accuracy","top_k_categorical_accuracy"])
    model.summary()

    return model


def build_lr_schedule(
        lr_max: float,
        lr_warmup_epochs: int,
        lr_sustain_epochs: int,
        lr_decay: float,
    ):
    def get_lr(epoch: int):
        lr_min = lr_start = lr_max / 100
        if epoch < lr_warmup_epochs:
            lr = (lr_max - lr_start) / lr_warmup_epochs * epoch + lr_start
        elif epoch < lr_warmup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = ((lr_max - lr_min) *
                   lr_decay ** (epoch - lr_warmup_epochs - lr_sustain_epochs) +
                   lr_min)
        return lr
    return get_lr


def get_strategy():
    try:
        # No parameters necessary if TPU_NAME environment variable is set.
        # On Kaggle this is always the case.
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy() # works on CPU and single GPU

    print(f'num replicas: {strategy.num_replicas_in_sync}, strategy {strategy}')
    return strategy, tpu


# def setup_policy(mixed_precision: bool, xla_accelerate: bool, tpu: bool):
#     if mixed_precision:
#         policy = tf.keras.mixed_precision.experimental.Policy(
#             'mixed_bfloat16' if tpu else 'mixed_float16')
#         tf.keras.mixed_precision.experimental.set_policy(policy)
#         print(f'Mixed precision enabled: {policy}')

#     if xla_accelerate:
#         tf.config.optimizer.set_jit(True)
#         print('XLA enabled')


if __name__ == '__main__':
    main()
