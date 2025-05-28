import tensorflow as tf
import numpy as np

def positional_encoding(length, d_model):
    pos = np.arange(length)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_encoding = tf.Variable(positional_encoding(max_len, d_model), trainable=False)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

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
        attn_output = self.mha(x, x, x, attention_mask=mask, training=training)
        out1 = self.layernorm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + self.dropout2(ffn_output, training=training))
        return out2

class build_ctc_transformer_model(tf.keras.Model):
    def __init__(
        self,
        input_dim,
        vocab_size,
        max_len=1000,
        d_model=128,
        num_heads=4,
        num_layers=4,
        ff_dim=256,
        dropout_rate=0.1
    ):
        super().__init__()
        self.input_proj = tf.keras.layers.Dense(d_model)
        self.pos_encoding = PositionalEncodingLayer(max_len, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ]
        self.output_dense = tf.keras.layers.Dense(vocab_size + 1)  # +1 for CTC blank

    def compute_mask(self, inputs):
        # Tạo mask dạng (B, T)
        mask = tf.reduce_any(tf.not_equal(inputs, 0.0), axis=-1)
        # Đưa về dạng attention mask (B, 1, 1, T)
        return tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype=tf.float32)

    def call(self, inputs, training=False):
        x = self.input_proj(inputs)
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        mask = self.compute_mask(inputs)

        for layer in self.encoder_layers:
            x = layer(x, training=training, mask=mask)

        return self.output_dense(x)
