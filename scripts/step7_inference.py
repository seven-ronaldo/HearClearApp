# step7_inference.py
import tensorflow as tf
import numpy as np
import librosa
import sys
from step2_dataset import load_char2idx

N_MELS = 128
SAMPLE_RATE = 16000  # Chuẩn sample rate của data
MAX_TIME_STEPS = 300  # Giới hạn max time steps (tùy chỉnh)

def preprocess_wav(file_path):
    # Load wav, convert mono 16kHz
    wav, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    # Tính log-mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=512,
        hop_length=160,
        n_mels=N_MELS,
        power=1.0
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    # chuẩn hóa log_mel_spec về 0~1 (cách riêng)
    norm_log_mel = (log_mel_spec + 80) / 80  # giả định dB trong [-80, 0]
    # transpose về (time, n_mels)
    norm_log_mel = norm_log_mel.T

    # cắt hoặc padding về MAX_TIME_STEPS
    if norm_log_mel.shape[0] > MAX_TIME_STEPS:
        norm_log_mel = norm_log_mel[:MAX_TIME_STEPS, :]
    else:
        pad_len = MAX_TIME_STEPS - norm_log_mel.shape[0]
        norm_log_mel = np.pad(norm_log_mel, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)

    # shape: (1, time, n_mels)
    return np.expand_dims(norm_log_mel.astype(np.float32), axis=0)

def decode_predictions(logits, char_map):
    y_pred = tf.nn.softmax(logits, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1).numpy()
    idx2char = {v: k for k, v in char_map.items()}

    decoded = []
    prev = -1
    for idx in y_pred[0]:
        if idx != prev and idx != 0:
            decoded.append(idx2char.get(idx, ""))
        prev = idx
    return "".join(decoded)

def run_inference(tflite_path, wav_path):
    char2idx = load_char2idx()

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = preprocess_wav(wav_path)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Lấy output logits
    output_data = interpreter.get_tensor(output_details[0]['index'])

    pred_text = decode_predictions(output_data, char2idx)

    print(f"🗣️ File: {wav_path}")
    print(f"📝 Dự đoán: {pred_text}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python step7_inference.py model.tflite file.wav")
        sys.exit(1)
    tflite_path = sys.argv[1]
    wav_path = sys.argv[2]
    run_inference(tflite_path, wav_path)
