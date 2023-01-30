import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, ReLU, AveragePooling2D, ZeroPadding2D

class keras_fpa(tf.keras.layers.Layer):
    def __init__(self, channels=2048, output_channels=256):
        """
        Module of Feature Pyramid Attention
        :type channels: int
        """
        super(keras_fpa, self).__init__()

        self.input_channels = channels
        self.output_channels = output_channels

        channels_mid = int(self.input_channels/4)

        # Master branch
        self.conv_master = Conv2D(self.input_channels, kernel_size=1, use_bias=False)
        self.bn_master = BatchNormalization(epsilon=1e-5, momentum=0.1)

        # Global pooling branch
        self.conv_gpb = Conv2D(self.input_channels, kernel_size=1, use_bias=False)
        self.bn_gpb = BatchNormalization(epsilon=1e-5, momentum=0.1)

        # C333 because of the shape of last feature maps is (16, 16).
        self.conv7x7_1 = Conv2D(channels_mid, kernel_size=(7, 7), strides=2, use_bias=False)
        self.bn1_1 = BatchNormalization(epsilon=1e-5, momentum=0.1)
        self.conv7x7_2 = Conv2D(channels_mid, kernel_size=(7, 7), strides=1, use_bias=False)
        self.bn1_2 = BatchNormalization(epsilon=1e-5, momentum=0.1)

        self.conv5x5_1 = Conv2D(channels_mid, kernel_size=(5, 5), strides=2, use_bias=False)
        self.bn2_1 = BatchNormalization(epsilon=1e-5, momentum=0.1)
        self.conv5x5_2 = Conv2D(channels_mid, kernel_size=(5, 5), strides=1, use_bias=False)
        self.bn2_2 = BatchNormalization(epsilon=1e-5, momentum=0.1)

        self.conv3x3_1 = Conv2D(channels_mid, kernel_size=(3, 3), strides=2, use_bias=False)
        self.bn3_1 = BatchNormalization(epsilon=1e-5, momentum=0.1)
        self.conv3x3_2 = Conv2D(channels_mid, kernel_size=(3, 3), strides=1, use_bias=False)
        self.bn3_2 = BatchNormalization(epsilon=1e-5, momentum=0.1)

        # Convolution Upsample
        self.conv_upsample_3 = Conv2DTranspose(channels_mid, kernel_size=4, strides=2, padding="same", use_bias=False)
        self.bn_upsample_3 = BatchNormalization(epsilon=1e-5, momentum=0.1)

        self.conv_upsample_23 = Conv2DTranspose(channels_mid, kernel_size=4, strides=2, padding="same", use_bias=False)
        self.bn_upsample_23 = BatchNormalization(epsilon=1e-5, momentum=0.1)

        self.conv_upsample_1 = Conv2DTranspose(self.input_channels, kernel_size=4, strides=2, padding="same", use_bias=False)
        self.bn_upsample_1 = BatchNormalization(epsilon=1e-5, momentum=0.1)

        self.relu = ReLU()

        self.conv_final = Conv2D(output_channels, kernel_size=1, strides=1, use_bias=False)

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.input_channels,
            "output_channels": self.output_channels
        })
        return config

    def call(self, x):
        """
        :param x: Shape: [m, h, w, 2048]
        :return: out: Feature maps. Shape: [m, h, w, 2048]
        """

        #Pooling branch for global pooling
        x_gpb = AveragePooling2D(x.shape[1:3])(x)
        x_gpb = self.conv_gpb(x_gpb)
        x_gpb = self.bn_gpb(x_gpb)
        # print("x_gpb", x_gpb.shape)

        # Master branch
        x_master = self.conv_master(x)
        x_master = self.bn_master(x_master)
        # print("x_master", x_master.shape)

        # Branch 1 (two 7x7 convolutions)
        x1_1 = ZeroPadding2D(padding=3)(x)
        x1_1 = self.conv7x7_1(x1_1)
        x1_1 = self.bn1_1(x1_1)
        x1_1 = self.relu(x1_1)
        x1_2 = ZeroPadding2D(padding=3)(x1_1)
        x1_2 = self.conv7x7_2(x1_2)
        x1_2 = self.bn1_2(x1_2)
        # print("x1_1", x1_1.shape)
        # print("x1_2", x1_2.shape)

        # Branch 2 (two 5x5 convolutions)
        x2_1 = ZeroPadding2D(padding=2)(x1_1)
        x2_1 = self.conv5x5_1(x2_1)
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu(x2_1)
        x2_2 = ZeroPadding2D(padding=2)(x2_1)
        x2_2 = self.conv5x5_2(x2_2)
        x2_2 = self.bn2_2(x2_2)
        # print("x2_1", x2_1.shape)
        # print("x2_2", x2_2.shape)

        # Branch 3 (two 3x3 convolutions)
        x3_1 = ZeroPadding2D(padding=1)(x2_1)
        x3_1 = self.conv3x3_1(x3_1)
        x3_1 = self.bn3_1(x3_1)
        x3_1 = self.relu(x3_1)
        x3_2 = ZeroPadding2D(padding=1)(x3_1)
        x3_2 = self.conv3x3_2(x3_2)
        x3_2 = self.bn3_2(x3_2)
        # print("x3_1", x3_1.shape)
        # print("x3_2", x3_2.shape)

        # To make the feature maps with mutiple scales, we firstly need
        # to merge results from the three branches below, then multiplied
        # with features maps
        x3_upsample = self.relu(
            self.bn_upsample_3(
                self.conv_upsample_3(x3_2)))
        # print("x3_upsample", x3_upsample.shape)

        x23_merge = self.relu(
            x2_2 + x3_upsample) #merging branch 2 and branch 3
        # print("x23_merge", x3_upsample.shape)


        x23_upsample = self.relu(
            self.bn_upsample_23(
                self.conv_upsample_23(x23_merge)))
        # print("x23_upsample", x23_upsample.shape)

        # x123_merge contains the feature maps from the three branches
        x123_merge = self.relu(x1_2 + x23_upsample)
        # print("x123_merge", x123_merge.shape)

        x123_upsample = self.bn_upsample_1(
                self.conv_upsample_1(x123_merge))
        # print("x123_upsample", x123_upsample.shape)
        
        # multiplied with feature maps after after a 1 × 1 convolution
        x_integrated_with_mutiple_scales = x_master * self.relu(x123_upsample)
        # print("x_integrated_with_mutiple_scales", x_integrated_with_mutiple_scales.shape)

        # integrate all the results (features maps,features maps integrated
        # with mutiple scales,and global attention)
        out = self.relu(x + x_gpb + x_integrated_with_mutiple_scales)
        # print("out", out.shape)

        f = self.conv_final(out)

        return f
