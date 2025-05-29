import tensorflow as tf
import numpy as np
import librosa
import sys
import os
import traceback
import logging

from step2_dataset import load_char2idx

# ======== Tiền xử lý âm thanh ========
def load_audio_mfcc(file_path, sample_rate=16000, n_mfcc=128, fixed_length=112):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T  # (T, n_mfcc)
    if mfcc.shape[0] > fixed_length:
        mfcc = mfcc[:fixed_length, :]
    else:
        pad_width = fixed_length - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    mfcc = mfcc.astype(np.float32)
    mfcc = np.expand_dims(mfcc, axis=0)  # (1, fixed_length, n_mfcc)
    return mfcc

# ======== Hàm Greedy CTC decoding ========
def ctc_greedy_decode(logits, blank_index=0):
    # logits shape: (1, T, vocab_size)
    pred_ids = np.argmax(logits, axis=-1)[0]  # lấy sequence đầu tiên (T,)
    deduped = []
    prev = blank_index
    for idx in pred_ids:
        if idx != prev and idx != blank_index:
            deduped.append(idx)
        prev = idx
    return [deduped]  # danh sách 1 sequence

# ======== Decode từ ID về văn bản ========
def decode_to_text(decoded_ids, idx2char):
    if len(decoded_ids) == 0:
        return ""
    ids = decoded_ids[0]
    chars = [idx2char.get(i, '') for i in ids]
    return ''.join(chars)

# ======== Chạy suy luận với mô hình TFLite ========
def run_inference(tflite_model_path, wav_path):
    try:
        print("🔍 Load mô hình TFLite...")
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"📥 Input tensor info: {input_details[0]}")
        print(f"📤 Output tensor info: {output_details[0]}")

        print("🎧 Tiền xử lý file âm thanh...")
        mfcc = load_audio_mfcc(wav_path, n_mfcc=128, fixed_length=112)  # (1, T, n_mfcc)

        # Kiểm tra xem input model có shape phù hợp
        expected_shape = input_details[0]['shape']
        print(f"Expected input shape: {expected_shape}")
        print(f"Actual MFCC input shape: {mfcc.shape}")

        # Nếu chiều T (frame length) model chờ cố định, cần reshape hoặc cắt/dãn
        if expected_shape[1] != -1 and expected_shape[1] != mfcc.shape[1]:
            # Nếu model input cố định frame length, cần xử lý ở đây
            # Ví dụ cắt hoặc padding mfcc về đúng chiều expected_shape[1]
            target_len = expected_shape[1]
            current_len = mfcc.shape[1]
            if current_len > target_len:
                mfcc = mfcc[:, :target_len, :]
            else:
                pad_len = target_len - current_len
                mfcc = np.pad(mfcc, ((0,0), (0,pad_len), (0,0)), mode='constant')
            print(f"MFCC input đã được reshape/padding về: {mfcc.shape}")

        # Đặt tensor input
        interpreter.set_tensor(input_details[0]['index'], mfcc)
        interpreter.invoke()

        logits = interpreter.get_tensor(output_details[0]['index'])  # (1, T, vocab_size)

        # Load từ điển và chuẩn bị decode
        char2idx = load_char2idx()
        idx2char = {v: k for k, v in char2idx.items()}
        blank_index = char2idx.get('<blank>', 0)

        # Decode logits thành văn bản
        decoded_ids = ctc_greedy_decode(logits, blank_index=blank_index)
        decoded_text = decode_to_text(decoded_ids, idx2char)

        print("📝 Văn bản nhận diện:")
        print(decoded_text)

    except Exception:
        print("❌ Lỗi khi chạy inference:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Cách dùng: python step7_inference.py <tflite_model_path> <wav_path>")
        sys.exit(1)

    tflite_model_path = sys.argv[1]
    wav_path = sys.argv[2]

    if not os.path.exists(tflite_model_path):
        print(f"❌ File mô hình không tồn tại: {tflite_model_path}")
        sys.exit(1)
    if not os.path.exists(wav_path):
        print(f"❌ File âm thanh không tồn tại: {wav_path}")
        sys.exit(1)

    run_inference(tflite_model_path, wav_path)
