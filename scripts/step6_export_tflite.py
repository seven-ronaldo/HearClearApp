import tensorflow as tf
import numpy as np
import os
import sys
import traceback
import logging

from step3_model import build_ctc_transformer_model
from step2_dataset import load_char2idx
from step4_train import CTCModel  # Wrapper đã train

def export_tflite():
    try:
        # Tắt logging của TensorFlow ở mức ERROR trở lên, loại bỏ nhiều thông báo verbose
        tf.get_logger().setLevel(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 3 = chỉ lỗi nghiêm trọng

        print("🔧 Bắt đầu export TFLite...")

        # Load vocab size
        char2idx = load_char2idx()
        vocab_size = len(char2idx)
        print(f"✅ Loaded vocab size: {vocab_size}")

        # Build base model
        base_model = build_ctc_transformer_model(
            input_dim=128,
            vocab_size=vocab_size,
            d_model=128,
            num_heads=4,
            num_layers=4,
            ff_dim=256,
        )
        model = CTCModel(base_model)

        # Build với dummy input
        dummy_input = tf.random.uniform((1, 1000, 128))
        _ = model(dummy_input, training=False)

        # Load weights
        model_path = "checkpoints/best_model.weights.h5"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Không tìm thấy file weights: {model_path}")
        model.load_weights(model_path)
        print(f"✅ Loaded weights từ {model_path}")

        # Tách base model ra khỏi wrapper
        base_model = model.model

        # Convert sang TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        print("⏳ Đang convert sang TFLite...")
        try:
            tflite_model = converter.convert()
        except Exception as e:
            print("❌ Lỗi khi convert TFLite:")
            traceback.print_exc()
            return

        print("✅ Convert thành công.")

        # Lưu file
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
