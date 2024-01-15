import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix


class UNetModel:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        inputs = tf.keras.layers.Input(shape=[128, 128, 3])
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=[128, 128, 3], include_top=False
        )

        layer_names = [
            "block_1_expand_relu",  # 64x64
            "block_3_expand_relu",  # 32x32
            "block_6_expand_relu",  # 16x16
            "block_13_expand_relu",  # 8x8
            "block_16_project",  # 4x4
        ]
        base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

        down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

        down_stack.trainable = False

        up_stack = [
            pix2pix.upsample(512, 3),  # 4x4 -> 8x8
            pix2pix.upsample(256, 3),  # 8x8 -> 16x16
            pix2pix.upsample(128, 3),  # 16x16 -> 32x32
            pix2pix.upsample(64, 3),  # 32x32 -> 64x64
        ]

        skips = down_stack(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])

        for up, skip in zip(up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        last = tf.keras.layers.Conv2DTranspose(
            filters=1, kernel_size=3, strides=2, padding="same", activation="sigmoid"
        )

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    def compile(self):
        self.model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

    def train(self, X_train, y_train, X_val, y_val, epochs=20):
        return self.model.fit(
            X_train, y_train, validation_data=(X_val, y_val), epochs=epochs
        )

    def predict(self, X):
        return self.model.predict(X)
