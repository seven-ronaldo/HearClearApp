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
    pos_encoding = angle_rads[np.newaxis, ...]  # shape (1, length, d_model)
    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_encoding = positional_encoding(max_len, d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            name="multi_head_attention"
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu', name="ffn_dense1"),
            tf.keras.layers.Dense(d_model, name="ffn_dense2")
        ], name="ffn")
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layernorm1")
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layernorm2")
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate, name="dropout1")
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate, name="dropout2")

    def call(self, x, training=None, mask=None):
        # mask cần dạng bool hoặc float 0/1 với shape broadcast được
        attn_output = self.mha(x, x, x, attention_mask=mask, training=training)
        out1 = self.layernorm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + self.dropout2(ffn_output, training=training))
        return out2

def build_ctc_transformer_model(
    input_dim,
    vocab_size,
    max_len=1000,
    d_model=128,
    num_heads=4,
    num_layers=4,
    ff_dim=256,
    dropout_rate=0.1,
):
    inputs = tf.keras.Input(shape=(None, input_dim), name="input")  # (B, T, F)

    # Tạo mask (B, T) bằng Lambda layer
    mask = tf.keras.layers.Lambda(
        lambda x: tf.reduce_any(tf.not_equal(x, 0.0), axis=-1),
        name="padding_mask"
    )(inputs)  # (B, T), bool

    # Convert mask thành attention mask dạng (B, 1, 1, T) float
    attention_mask = tf.keras.layers.Lambda(
        lambda x: tf.cast(x[:, tf.newaxis, tf.newaxis, :], dtype=tf.float32),
        name="attention_mask"
    )(mask)

    x = tf.keras.layers.Dense(d_model, name="input_projection")(inputs)
    x = PositionalEncodingLayer(max_len, d_model)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    for i in range(num_layers):
        x = TransformerEncoderLayer(d_model, num_heads, ff_dim, dropout_rate)(x, training=True, mask=attention_mask)

    logits = tf.keras.layers.Dense(vocab_size + 1, name="logits")(x)  # +1 for CTC blank token

    return tf.keras.Model(inputs=inputs, outputs=logits)


