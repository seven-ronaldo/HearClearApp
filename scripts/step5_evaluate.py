# step5_evaluate.py : Đánh giá mô hình
import os
import datetime
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import jiwer
import pandas as pd

from step3_model import build_ctc_transformer_model
from step4_train import CTCModel
from step2_dataset import (
    create_dataset_from_index_list,
    get_data_index_list,
    load_char2idx
)

# ========== Decode ==========

def decode_predictions(logits, char_map):
    y_pred = tf.nn.softmax(logits, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1).numpy()
    idx2char = {v: k for k, v in char_map.items()}

    results = []
    for seq in y_pred:
        prev = -1
        decoded = []
        for idx in seq:
            if idx != prev and idx != 0:
                decoded.append(idx2char.get(idx, ""))
            prev = idx
        results.append("".join(decoded))
    return results

def decode_label(label_seq, length, idx2char):
    return "".join([idx2char.get(idx, "") for idx in label_seq[:length] if idx != 0])

# ========== Evaluation ==========

def evaluate_model():
    print("🚀 Bắt đầu đánh giá mô hình...")

    # ===== Load dữ liệu =====
    all_indices = get_data_index_list('processed_chunks')
    total_samples = len(all_indices)
    val_indices = all_indices[int(0.9 * total_samples):]

    val_dataset = create_dataset_from_index_list(
        val_indices,
        batch_size=8,
        is_training=False
    )

    # ===== Load từ điển =====
    char2idx = load_char2idx()
    idx2char = {v: k for k, v in char2idx.items()}
    vocab_size = len(char2idx)

    # ===== Build mô hình gốc =====
    base_model = build_ctc_transformer_model(
        input_dim=128,
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=4,
        ff_dim=256
    )
    # Build base_model
    dummy_input = tf.random.normal([1, 100, 128])
    base_model(dummy_input)

    # ===== Tạo CTC Model =====
    model = CTCModel(base_model)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))

    # ✅ Build model đúng cách
    model(tf.random.normal([1, 100, 128]))

    # ===== Load weights =====
    model_path = "checkpoints/best_model.weights.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Không tìm thấy weights: {model_path}")
    
    model.load_weights(model_path)
    print(f"✅ Đã load weights từ {model_path}")

    # ===== Đánh giá =====
    val_loss = 0.0
    preds_all, gts_all = [], []

    for batch in val_dataset:
        features, labels, input_len, label_len = batch

        # Tính loss
        loss_dict = model.test_step(batch)
        val_loss += loss_dict["val_loss"].numpy()

        # Dự đoán
        logits = base_model(features, training=False)
        pred_texts = decode_predictions(logits, char2idx)

        label_texts = [
            decode_label(l.numpy(), l_len.numpy(), idx2char)
            for l, l_len in zip(labels, label_len)
        ]

        preds_all.extend(pred_texts)
        gts_all.extend(label_texts)

        if len(preds_all) >= 100:
            break

    avg_loss = val_loss / (len(preds_all) / 8)
    wer = jiwer.wer(gts_all, preds_all)
    cer = jiwer.cer(gts_all, preds_all)

    print(f"\n✅ Validation Loss: {avg_loss:.4f}")
    print(f"📊 WER: {wer:.4f} | CER: {cer:.4f}\n")

    # ===== Lưu kết quả =====
    result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "val_loss": float(round(avg_loss, 4)),
        "wer": float(round(wer, 4)),
        "cer": float(round(cer, 4)),
        "samples": len(preds_all),
        "model_path": model_path
    }

    with open("eval_history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("📁 Evaluation result appended to eval_history.jsonl ✅")
    return result

# ========== Plotting ==========

def plot_eval_history(jsonl_path="eval_history.jsonl"):
    if not os.path.exists(jsonl_path):
        print("⚠️ Chưa có file lịch sử đánh giá.")
        return

    df = pd.read_json(jsonl_path, lines=True)
    if len(df) < 2:
        print("ℹ️ Cần ít nhất 2 lần đánh giá để vẽ biểu đồ.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    plt.figure(figsize=(10, 5))
    plt.plot(df["timestamp"], df["wer"], label="WER", marker='o')
    plt.plot(df["timestamp"], df["cer"], label="CER", marker='s')
    plt.xlabel("Thời gian")
    plt.ylabel("Tỉ lệ lỗi")
    plt.title("📈 Biểu đồ WER / CER theo thời gian đánh giá")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("eval_plot.png")
    print("📊 Đã lưu biểu đồ tại eval_plot.png ✅")

# ========== MAIN ==========

if __name__ == "__main__":
    evaluate_model()
    plot_eval_history()
