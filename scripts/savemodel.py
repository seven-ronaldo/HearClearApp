from step3_model import build_ctc_transformer_model
from step2_dataset import load_char2idx
from step4_train import CTCModel
import tensorflow as tf
import os

# Build lại model như cũ
char2idx = load_char2idx()
vocab_size = len(char2idx)

base_model = build_ctc_transformer_model(
    input_dim=128,
    vocab_size=vocab_size,
    d_model=128,
    num_heads=4,
    num_layers=4,
    ff_dim=256,
)

ctc_model = CTCModel(base_model)
dummy_input = tf.random.uniform((1, 1000, 128))
_ = ctc_model(dummy_input)
ctc_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

# Load lại best weights
ctc_model.load_weights("checkpoints/best_model.weights.h5")
print("✅ Đã load weights tốt nhất")

# Export lại full model
os.makedirs("exported", exist_ok=True)
tf.saved_model.save(ctc_model.model, "exported/full_model")
print("✅ Đã lưu mô hình dạng SavedModel vào exported/full_model")
