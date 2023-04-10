import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from vit_keras import MultiHeadAttention

# Swin Transformer Block
class SwinTransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, droprate=0.0):
        super(SwinTransformerBlock, self).__init__()

        # Multi-Head Self-Attention
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=droprate,
        )

        # First Layer Norm
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)

        # Feedforward Layer
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation="gelu"),
            layers.Dropout(droprate),
            layers.Dense(embed_dim),
            layers.Dropout(droprate),
        ])

        # Second Layer Norm
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training):
        # Multi-Head Self-Attention
        attn_output = self.attention(inputs, inputs)
        attn_output = layers.Dropout(0.1)(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)

        # Feedforward Layer
        mlp_output = self.mlp(out1)
        out2 = self.norm2(out1 + mlp_output)

        return out2

# SwinUNet Model with Monte Carlo Dropout
class SwinUNet(tf.keras.Model):
    def __init__(self, in_channels=1, n_classes=5, embed_dim=96, num_heads=4, num_layers=4, mlp_dim=384, droprate=0.2, num_samples=50):
        super(SwinUNet, self).__init__()

        # Downsampling Path
        self.downs = []
        for i in range(num_layers):
            # Downsampling Conv Block
            self.downs.append(tf.keras.Sequential([
                layers.Conv2D(embed_dim*(2**i), 3, padding="same"),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Conv2D(embed_dim*(2**i), 3, padding="same"),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.MaxPooling2D(pool_size=2, strides=2)
            ]))
            # Swin Transformer Blocks
            for j in range(2):
                self.downs.append(SwinTransformerBlock(embed_dim*(2**i), num_heads, mlp_dim*(2**i), droprate))

        # Bottleneck Conv Block
        self.bottleneck = tf.keras.Sequential([
            layers.Conv2D(embed_dim*(2**num_layers), 3, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv2D(embed_dim*(2**num_layers), 3, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])

        # Upsampling Path
        self.ups = []
        for i in range(num_layers-1, -1, -1):
            # Upsampling Conv Block
            self.ups.append(tf.keras.Sequential([
                layers.Conv2DTranspose(embed_dim*(2**i), kernel_size=2, strides=2),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Conv2D(embed_dim*(2**i), 3, padding="same"),
                layers.BatchNormalization(),
                layers.Activation("relu"),            ]))
            # Swin Transformer Blocks
            for j in range(2):
                self.ups.append(SwinTransformerBlock(embed_dim*(2**i), num_heads, mlp_dim*(2**i), droprate))

        # Output Conv Block
        self.out = layers.Conv2D(n_classes, kernel_size=1)

        # Monte Carlo Dropout
        self.mc_dropout = layers.Dropout(droprate)
        self.num_samples = num_samples

    def call(self, inputs, training):
        # Downsampling Path
        skips = []
        x = inputs
        for down in self.downs:
            if isinstance(down, SwinTransformerBlock):
                x = down(x, training=training)
            else:
                x = down(x)
                skips.append(x)
        skips = reversed(skips[:-1])

        # Bottleneck Conv Block
        x = self.bottleneck(x)

        # Upsampling Path
        for up, skip in zip(self.ups, skips):
            if isinstance(up, SwinTransformerBlock):
                x = up(x, training=training)
            else:
                x = up(x)
                x = tf.concat([x, skip], axis=-1)

        # Output Conv Block
        x = self.out(x)

        # Monte Carlo Dropout
        if training:
            x = self.mc_dropout(x, training=True)

        return x

    # Function to predict with Monte Carlo Dropout
    def predict_mc(self, dataset):
        pred_list = []
        for i in range(self.num_samples):
            pred = []
            for x, y in dataset:
                y_pred = self(x, training=True)
                pred.append(y_pred.numpy())
            pred_list.append(pred)
        return np.array(pred_list)

               
