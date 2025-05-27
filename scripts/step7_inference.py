# step7_inference.py

import tensorflow as tf
import numpy as np
import librosa
import os
import sys
import traceback

from step2_dataset import load_char2idx

# --- Cấu hình ---
N_MELS = 128
SAMPLE_RATE = 16000
MAX_TIME_STEPS = 300

def preprocess_wav(file_path):
    wav, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mel_spec = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=512,
        hop_length=160,
        n_mels=N_MELS,
        power=1.0
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    norm_log_mel = (log_mel_spec + 80) / 80  # normalize to 0-1
    norm_log_mel = norm_log_mel.T

    # Pad / crop
    if norm_log_mel.shape[0] > MAX_TIME_STEPS:
        norm_log_mel = norm_log_mel[:MAX_TIME_STEPS, :]
    else:
        pad_len = MAX_TIME_STEPS - norm_log_mel.shape[0]
        norm_log_mel = np.pad(norm_log_mel, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)

    return np.expand_dims(norm_log_mel.astype(np.float32), axis=0)  # (1, T, 128)

def greedy_ctc_decode(logits, blank_index):
    pred = np.argmax(logits, axis=-1)  # (B, T)
    results = []
    for seq in pred:
        decoded = []
        prev = -1
        for idx in seq:
            if idx != prev and idx != blank_index:
                decoded.append(idx)
            prev = idx
        results.append(decoded)
    return results

def decode_to_text(indices, idx2char):
    return ["".join([idx2char[i] for i in seq if i in idx2char]) for seq in indices]

def run_inference(tflite_path, wav_path):
    try:
        if not os.path.exists(tflite_path):
            raise FileNotFoundError(f"Không tìm thấy model: {tflite_path}")
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Không tìm thấy file wav: {wav_path}")

        print("🔍 Load vocab...")
        char2idx = load_char2idx()
        idx2char = {v: k for k, v in char2idx.items()}
        blank_index = len(idx2char)

        print("🔊 Tiền xử lý âm thanh...")
        input_data = preprocess_wav(wav_path)
        print(f"✅ Input shape: {input_data.shape}")

        print("📦 Load TFLite model...")
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.resize_tensor_input(input_details[0]['index'], input_data.shape)
        interpreter.allocate_tensors()

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        print(f"✅ Output shape: {output.shape}")

        decoded_indices = greedy_ctc_decode(output, blank_index)
        decoded_text = decode_to_text(decoded_indices, idx2char)

        print(f"🗣️ File: {wav_path}")
        print(f"📝 Dự đoán: {decoded_text[0]}")

    except Exception as e:
        print("❌ Lỗi khi chạy inference:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python step7_inference.py model.tflite file.wav")
        sys.exit(1)
    tflite_path = sys.argv[1]
    wav_path = sys.argv[2]
    run_inference(tflite_path, wav_path)

