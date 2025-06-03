# step6_export_tflite.py

import tensorflow as tf
import numpy as np
import os
import sys
import traceback
import logging

from step3_model import build_ctc_transformer_model
from step2_dataset import load_char2idx
from step4_train import CTCModel


def export_tflite():
    try:
        # Tắt cảnh báo log
        tf.get_logger().setLevel(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        print("🔧 Bắt đầu export mô hình sang TensorFlow Lite...")

        # Load vocab
        print("📚 Load vocabulary...")
        char2idx = load_char2idx()
        vocab_size = len(char2idx)
        print(f"✅ Vocabulary size: {vocab_size}")

        # Khởi tạo mô hình gốc
        print("🧠 Khởi tạo mô hình...")
        base_model = build_ctc_transformer_model(
            input_dim=128,
            vocab_size=vocab_size,
            d_model=128,
            num_heads=4,
            num_layers=4,
            ff_dim=256,
        )

        # Gói base_model vào CTCModel để load đúng weights
        ctc_model = CTCModel(base_model)

        # Build model bằng dummy input
        dummy_input = tf.random.uniform((1, 1000, 128), dtype=tf.float32)
        _ = ctc_model(dummy_input, training=False)

        # Load weights
        model_path = "checkpoints/best_model.weights.h5"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Không tìm thấy file weights: {model_path}")
        ctc_model.load_weights(model_path)
        print(f"✅ Đã load weights từ: {model_path}")

        # Lấy lại base_model để export
        trained_base_model = ctc_model.model
        inputs = trained_base_model.input
        logits = trained_base_model.output
        softmax_output = tf.keras.layers.Activation('softmax', name='softmax_output')(logits)
        export_model = tf.keras.Model(inputs=inputs, outputs=softmax_output)

        # Gọi model với dummy input để khởi tạo
        _ = export_model(dummy_input, training=False)

        # Tạo concrete function để export
        print("🔨 Tạo concrete function để export...")
        @tf.function(input_signature=[tf.TensorSpec(shape=[1, 1000, 128], dtype=tf.float32)])
        def inference_fn(inputs):
            return export_model(inputs, training=False)

        concrete_func = inference_fn.get_concrete_function()

        # Convert sang TFLite
        print("🔄 Đang chuyển đổi sang định dạng .tflite...")
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.experimental_enable_resource_variables = False
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Bật nếu cần

        tflite_model = converter.convert()
        print("✅ Chuyển đổi thành công!")

        # Lưu model ra file
        os.makedirs("tflite_model", exist_ok=True)
        tflite_path = "tflite_model/model.tflite"
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"💾 Mô hình đã được lưu tại: {tflite_path}")

        # Kiểm tra lại mô hình TFLite
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        print("🔍 Kiểm tra mô hình TFLite thành công.")

    except Exception:
        print("❌ Gặp lỗi trong quá trình export:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    export_tflite()
