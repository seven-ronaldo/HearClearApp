import tensorflow as tf
import numpy as np
import os
import sys
import traceback
import logging

from step3_model import build_ctc_transformer_model
from step2_dataset import load_char2idx
from step4_train import CTCModel  # Wrapper đã train


class CTCWrapperModule(tf.Module):
    def __init__(self, keras_model):
        super().__init__()
        self.model = keras_model

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, None, 128], dtype=tf.float32, name="input")])
    def __call__(self, inputs):
        logits = self.model(inputs, training=False)  # (1, T, vocab_size + 1)
        input_len = tf.shape(logits)[1]
        input_len = tf.expand_dims(input_len, 0)  # (1,)

        # Transpose để dùng cho CTC decode: (T, B, C)
        decoded, _ = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(logits, [1, 0, 2]),
            sequence_length=input_len
        )
        dense_decoded = tf.sparse.to_dense(decoded[0], default_value=-1)  # (1, T_decoded)
        return {"char_ids": dense_decoded}


def export_tflite():
    try:
        # Disable TF logs
        tf.get_logger().setLevel(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        print("🔧 Bắt đầu export TFLite...")

        # Load vocab
        char2idx = load_char2idx()
        vocab_size = len(char2idx)
        print(f"✅ Loaded vocab size: {vocab_size}")

        # Build model
        base_model = build_ctc_transformer_model(
            input_dim=128,
            vocab_size=vocab_size,
            d_model=128,
            num_heads=4,
            num_layers=4,
            ff_dim=256,
        )
        model = CTCModel(base_model)

        # Thay vì dùng tf.random.uniform
        dummy_input = tf.ones((1, 1000, 128), dtype=tf.float32)
        _ = model(dummy_input, training=False)

        model_path = "checkpoints/best_model.weights.h5"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Không tìm thấy file weights: {model_path}")
        model.load_weights(model_path)
        print(f"✅ Loaded weights từ {model_path}")

        base_model = model.model
        wrapper_module = CTCWrapperModule(base_model)

        # Save as SavedModel
        export_dir = "exported/ctc_module"
        tf.saved_model.save(wrapper_module, export_dir=export_dir)
        print(f"✅ Đã lưu SavedModel tại {export_dir}")

        # Convert to TFLite
        print("⏳ Đang convert sang TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.allow_custom_ops = True  # Để tránh lỗi nếu CTC decode không hỗ trợ đầy đủ

        tflite_model = converter.convert()
        print("✅ Convert thành công.")

        os.makedirs("tflite_model", exist_ok=True)
        with open("tflite_model/model.tflite", "wb") as f:
            f.write(tflite_model)
        print("✅ Đã lưu model.tflite vào thư mục tflite_model/")

    except Exception as e:
        print("❌ Gặp lỗi khi export TFLite:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    export_tflite()
