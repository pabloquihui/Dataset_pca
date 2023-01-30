from keras.layers import *
from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, Activation, Dropout, AveragePooling2D, concatenate, GlobalAveragePooling2D, MaxPooling2D, Dense, Input
from keras.regularizers import l2
import keras.backend as K
from UQ.mcdropout import MCDropout
#############################################################
##              Attention
#############################################################
def attention_up_and_concate(down_layer, layer, data_format='channels_last'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4, data_format=data_format)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])
    return concate

# Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net)
def rec_res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],

                  padding='same', data_format='channels_last'):
    if data_format == 'channels_first':
        input_n_filters = input_layer.get_shape().as_list()[1]
    else:
        input_n_filters = input_layer.get_shape().as_list()[3]

    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            input_layer)
    else:
        skip_layer = input_layer

    layer = skip_layer
    for j in range(2):

        for i in range(2):
            if i == 0:

                layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                    layer)
                if batch_normalization:
                    layer1 = BatchNormalization()(layer1)
                layer1 = Activation('relu')(layer1)
            layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                add([layer1, layer]))
            if batch_normalization:
                layer1 = BatchNormalization()(layer1)
            layer1 = Activation('relu')(layer1)
        layer = layer1

    out_layer = add([layer, skip_layer])
    return out_layer


def attention_block_2d(x, g, inter_channel, data_format='channels_last'):
    # theta_x(?,g_height,g_width,inter_channel)

    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(x)

    # phi_g(?,g_height,g_width,inter_channel)

    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(g)

    # f(?,g_height,g_width,inter_channel)

    f = Activation('relu')(add([theta_x, phi_g]))

    # psi_f(?,g_height,g_width,1)

    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)

    rate = Activation('sigmoid')(psi_f)

    # rate(?,x_height,x_width)

    # att_x(?,x_height,x_width,x_channel)

    att_x = multiply([x, rate])

    return att_x


#################################################################
##                      DenseUnet
#################################################################

def Conv_Block(input_tensor, filters, bottleneck=False, weight_decay=1e-4):
    """    封装卷积层
    :param input_tensor: 输入张量
    :param filters: 卷积核数目
    :param bottleneck: 是否使用bottleneck
    :param dropout_rate: dropout比率
    :param weight_decay: 权重衰减率
    :return:
    """
    concat_axis = 1 if K.image_data_format() == 'channel_first' else -1  # 确定格式

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(input_tensor)
    x = Activation('relu')(x)

    # if bottleneck:
    #     # 使用bottleneck进行降维
    #     inter_channel = filters
    #     x = Conv2D(inter_channel, (1, 1),
    #                kernel_initializer='he_normal',
    #                padding='same', use_bias=False,
    #                kernel_regularizer=l2(weight_decay))(x)
    #     x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
    #     x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same', use_bias=False)(x)
    return x

def dens_block(input_tensor, nb_filter):
    x1 = Conv_Block(input_tensor,nb_filter)
    add1 = concatenate([x1, input_tensor], axis=-1)
    x2 = Conv_Block(add1,nb_filter)
    add2 = concatenate([x1, input_tensor,x2], axis=-1)
    x3 = Conv_Block(add2,nb_filter)
    return x3

def attn_dense_unet(input_shape=(256, 256, 1), num_classes = 5):
    data_format='channels_last'
    inputs = Input(input_shape)
    # x  = Conv2D(32, 1, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    x = Conv2D(16, (3,3), kernel_initializer='he_normal', padding='same', strides=1,use_bias=False, kernel_regularizer=l2(1e-4))(inputs)
    skips = []
    #down first
    down1 = dens_block(x,nb_filter=32)
    skips.append(down1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(down1)#256
    #down second
    down2 = dens_block(pool1,nb_filter=32)
    skips.append(down2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(down2)#128
    #down third
    down3 = dens_block(pool2,nb_filter=64)
    skips.append(down3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(down3)#64
    #down four
    down4 = dens_block(pool3,nb_filter=128)
    skips.append(down4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(down4)#32
    #center
    conv5 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)


    # up first
    # up6 = UpSampling2D(size=(2, 2))(drop5)
    # up6 = UpSampling2D(size=(2, 2))(drop5)
    up6 = attention_up_and_concate(drop5, skips[3], data_format=data_format)
    # add6 = concatenate([down4, up6], axis=3)
    up6 = dens_block(up6,nb_filter=128)

    # up second
    # up7 = UpSampling2D(size=(2, 2))(up6)
    #up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = attention_up_and_concate(up6, skips[2], data_format=data_format)
    # add7 = concatenate([down3, up7], axis=3)
    up7 = dens_block(up7,nb_filter=64)
    # up third
    # up8 = UpSampling2D(size=(2, 2))(up7)
    #up8 = UpSampling2D(size=(2, 2))(conv7)
    # add8 = concatenate([down2, up8], axis=-1)
    up8 = attention_up_and_concate(up7, skips[1], data_format=data_format)
    up8 = dens_block(up8,nb_filter=32)
    #up four
    # up9 =UpSampling2D(size=(2, 2))(up8)
    # add9 = concatenate([down1, up9], axis=-1)
    up9 = attention_up_and_concate(up8, skips[0], data_format=data_format)
    up9 = dens_block(up9,nb_filter=32)
    # output
    conv10 = Conv2D(16, (3,3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    conv10 = Conv2D(num_classes, (1,1), activation='softmax')(conv10)
    model = Model(inputs=inputs, outputs=conv10)
    return model

def mc_attn_dense_unet(input_shape=(256, 256, 1), num_classes = 5):
    data_format='channels_last'
    inputs = Input(input_shape)
    # x  = Conv2D(32, 1, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    x = Conv2D(16, (3,3), kernel_initializer='he_normal', padding='same', strides=1,use_bias=False, kernel_regularizer=l2(1e-4))(inputs)
    skips = []
    #down first
    down1 = dens_block(x,nb_filter=32)
    skips.append(down1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(down1)#256
    pool1 = MCDropout(0.1)(pool1)
    #down second
    down2 = dens_block(pool1,nb_filter=32)
    skips.append(down2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(down2)#128
    pool2 = MCDropout(0.1)(pool2)
    #down third
    down3 = dens_block(pool2,nb_filter=64)
    skips.append(down3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(down3)#64
    pool3 = MCDropout(0.2)(pool3)
    #down four
    down4 = dens_block(pool3,nb_filter=128)
    skips.append(down4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(down4)#32
    pool4 = MCDropout(0.2)(pool4)
    #center
    conv5 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = MCDropout(0.5)(conv5)


    # up first
    # up6 = UpSampling2D(size=(2, 2))(drop5)
    # up6 = UpSampling2D(size=(2, 2))(drop5)
    up6 = attention_up_and_concate(drop5, skips[3], data_format=data_format)
    # add6 = concatenate([down4, up6], axis=3)
    up6 = dens_block(up6,nb_filter=128)
    up6 = MCDropout(0.2)(up6)
    # up second
    # up7 = UpSampling2D(size=(2, 2))(up6)
    #up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = attention_up_and_concate(up6, skips[2], data_format=data_format)
    # add7 = concatenate([down3, up7], axis=3)
    up7 = dens_block(up7,nb_filter=64)
    up7 = MCDropout(0.2)(up7)
    # up third
    # up8 = UpSampling2D(size=(2, 2))(up7)
    #up8 = UpSampling2D(size=(2, 2))(conv7)
    # add8 = concatenate([down2, up8], axis=-1)
    up8 = attention_up_and_concate(up7, skips[1], data_format=data_format)
    up8 = dens_block(up8,nb_filter=32)
    up8 = MCDropout(0.1)(up8)
    #up four
    # up9 =UpSampling2D(size=(2, 2))(up8)
    # add9 = concatenate([down1, up9], axis=-1)
    up9 = attention_up_and_concate(up8, skips[0], data_format=data_format)
    up9 = dens_block(up9,nb_filter=32)
    up9 = MCDropout(0.1)(up9)
    # output
    conv10 = Conv2D(16, (3,3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    conv10 = Conv2D(num_classes, (1,1), activation='softmax')(conv10)
    model = Model(inputs=inputs, outputs=conv10)
    return model

if __name__ == "__main__":
    model = attn_dense_unet(input_shape=(256, 256, 1), num_classes = 5)
    model.compile(optimizer='Adam', loss=['categorical_crossentropy'], metrics=['accuracy'])
    model.summary()