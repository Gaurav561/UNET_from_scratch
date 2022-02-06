import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *


def UNET(shape):
    # def __init__(self,shape):
    #     self.shape = shape

    def encoder_block(input_to_conv,filters):
        c = Conv2D(filters,(3,3),activation = "relu",padding="same")(input_to_conv)
        x = Conv2D(filters,(3,3),activation = "relu",padding="same")(c)
        m = MaxPool2D((2,2))(x)

        print(x.shape)
        
        return x,m



    def decoder_block(input_to_upconv,conc_layer,filters):
        upconv = Conv2DTranspose(filters,(2,2),strides=(2,2))(input_to_upconv)
        conc = concatenate([upconv,conc_layer],axis=3)

        l1 = Conv2D(filters,(3,3),activation = "relu",padding="same")(conc)
        l2 = Conv2D(filters,(3,3),activation = "relu",padding="same")(l1)
        
        return l2

    def model():
        input_layer  = tf.keras.layers.Input(shape)
        x1 , m1 = encoder_block(input_layer,64)
        x2 , m2 = encoder_block(m1,128)
        x3 , m3 = encoder_block(m2,256)
        x4 , m4 = encoder_block(m3,512)
        x5 , m5 = encoder_block(m4,1024)

        y4 = decoder_block(x5,x4,512)
        y3 = decoder_block(y4,x3,256)
        y2 = decoder_block(y3,x2,128)
        y1 = decoder_block(y2,x1,64)

        output = Conv2D(2,(1,1))(y1)
    
        M = tf.keras.Model(input_layer,output)

        M.summary()

        return M

    Model  = model()

    return Model
    

Model = UNET((256,256,1))

# model = Model_class.model()


