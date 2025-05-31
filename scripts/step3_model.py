# Huấn luyện model

import tensorflow as tf
import numpy as np

def positional_encoding(length, d_model):
    pos = np.arange(length)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_encoding = positional_encoding(max_len, d_model)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False, mask=None):
        if mask is not None:
            # mask shape (B, T) --> (B, T, T)
            mask = tf.cast(mask, tf.bool)
            mask = tf.expand_dims(mask, axis=1)    # (B, 1, T)
            mask = tf.tile(mask, [1, tf.shape(mask)[-1], 1])  # (B, T, T)

        attn_output = self.mha(x, x, x, attention_mask=mask, training=training)
        out1 = self.layernorm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + self.dropout2(ffn_output, training=training))
        return out2

def build_ctc_transformer_model(input_dim, vocab_size, max_len=1000,
                                d_model=128, num_heads=4, num_layers=4, ff_dim=256):
    inputs = tf.keras.Input(shape=(None, input_dim), name="input")  # (B, T, input_dim)

    # Tạo mask: vị trí khác 0.0 => True, 0.0 => False
    mask = tf.keras.layers.Lambda(
        lambda x: tf.math.reduce_any(tf.math.not_equal(x, 0.0), axis=-1)
    )(inputs)  # (B, T), bool tensor

    x = tf.keras.layers.Dense(d_model)(inputs)
    x = PositionalEncodingLayer(max_len=max_len, d_model=d_model)(x)

    for _ in range(num_layers):
        x = TransformerEncoderLayer(d_model, num_heads, ff_dim)(x, training=True, mask=mask)

    logits = tf.keras.layers.Dense(vocab_size + 1, name="logits")(x)

    return tf.keras.Model(inputs=inputs, outputs=logits)