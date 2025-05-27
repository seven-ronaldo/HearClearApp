import tensorflow as tf
from step3_model import build_ctc_transformer_model
from step4_train import CTCModel
from step2_dataset import load_char2idx
import os

# === Load thông tin từ vocab để xác định input_dim, vocab_size ===
char2idx = load_char2idx()
vocab_size = len(char2idx)

# === Build lại mô hình ===
model = build_ctc_transformer_model(
    input_dim=128,      # MFCC features
    vocab_size=vocab_size,
    d_model=128,
    num_heads=4,
    num_layers=4,
    ff_dim=256
)

ctc_model = CTCModel(model)
ctc_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

# === Bắt buộc: Gọi dummy input để build model trước khi load weights ===
dummy_input = tf.random.uniform([1, 100, 128])  # batch=1, time_steps=100, feature_dim=128
ctc_model(dummy_input)

# === Load weights đã training xong ===
checkpoint_path = "checkpoints/best_model_full.keras"
ctc_model.load_weights(checkpoint_path)
print(f"✅ Loaded weights from {checkpoint_path}")

# === Save model ra định dạng .keras (dùng để load lại dễ dàng) ===
save_path = "saved_model/ctc_transformer_model_full.keras"
ctc_model.save(save_path)
print(f"✅ Saved model to {save_path}")

# === Export ra SavedModel format nếu muốn dùng cho TensorFlow Serving / TFLite ===
export_path = "saved_model/export_for_tflite"
os.makedirs(export_path, exist_ok=True)
ctc_model.export(export_path)
print(f"✅ Exported model for TFLite/Serving at {export_path}")
