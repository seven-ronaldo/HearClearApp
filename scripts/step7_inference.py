import tensorflow as tf
import numpy as np
import librosa
import sys
import os
import traceback
import re

from step2_dataset import load_char2idx  # Hàm load char2idx.npy chuẩn từ thư mục

def load_audio_mfcc(file_path, sample_rate=16000, n_mfcc=128, fixed_length=1000):
    """
    Load file audio, chuyển sang MFCC shape (1, fixed_length, n_mfcc).
    Nếu MFCC quá dài thì cắt, quá ngắn thì padding 0.
    """
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

def ctc_greedy_decode(logits, blank_index=0):
    """
    Giải mã logits bằng greedy CTC decode:
    - Chọn max mỗi step
    - Bỏ blank (index blank_index)
    - Collapse ký tự lặp liên tiếp
    """
    pred_ids = np.argmax(logits[0], axis=-1)  # logits shape: (1, time_steps, vocab_size)
    deduped = []
    prev = blank_index
    for idx in pred_ids:
        if idx != prev and idx != blank_index:
            deduped.append(idx)
        prev = idx
    return deduped  # Trả về danh sách index ký tự (không có blank và ký tự lặp)

def decode_to_text(decoded_ids, idx2char):
    """
    Chuyển danh sách index sang text:
    - Thay ký tự đặc biệt thành space nếu cần
    - Loại bỏ blank (nếu còn sót)
    - Chuẩn hóa khoảng trắng
    """
    chars = []
    for i in decoded_ids:
        ch = idx2char.get(i, '')
        if ch == '<blank>':
            continue
        if ch in ['_', ' ', 'SPACE']:
            ch = ' '
        chars.append(ch)
    text = ''.join(chars)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_inference(tflite_model_path, wav_path):
    try:
        print("🔍 Load mô hình TFLite...")
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"📥 Input tensor shape: {input_details[0]['shape']}")
        print(f"📤 Output tensor shape: {output_details[0]['shape']}")

        print("🎧 Tiền xử lý file âm thanh thành MFCC...")
        mfcc = load_audio_mfcc(wav_path, n_mfcc=input_details[0]['shape'][2], fixed_length=input_details[0]['shape'][1])
        print(f"MFCC shape sau tiền xử lý: {mfcc.shape}")

        # Đảm bảo MFCC đúng shape với input model
        expected_shape = input_details[0]['shape']
        if mfcc.shape != tuple(expected_shape):
            print(f"⚠️ Warning: MFCC shape {mfcc.shape} không khớp input model {expected_shape}, điều chỉnh lại...")
            current_len = mfcc.shape[1]
            target_len = expected_shape[1]
            if current_len > target_len:
                mfcc = mfcc[:, :target_len, :]
            else:
                pad_len = target_len - current_len
                mfcc = np.pad(mfcc, ((0,0),(0,pad_len),(0,0)), mode='constant')
            print(f"MFCC shape sau điều chỉnh: {mfcc.shape}")

        interpreter.set_tensor(input_details[0]['index'], mfcc)
        interpreter.invoke()

        logits = interpreter.get_tensor(output_details[0]['index'])  # shape (1, time_steps, vocab_size)
        print(f"Logits shape: {logits.shape}")

        # Load bảng char2idx và tạo idx2char
        char2idx = load_char2idx()
        idx2char = {v: k for k, v in char2idx.items()}
        blank_index = char2idx.get('<blank>', 0)

        print("🔤 Bảng idx2char sample (first 20 chars):")
        for i in range(min(20, len(idx2char))):
            print(f"  {i}: '{idx2char.get(i, '')}'")
        print(f"Blank index: {blank_index}")

        decoded_ids = ctc_greedy_decode(logits, blank_index=blank_index)
        decoded_text = decode_to_text(decoded_ids, idx2char)

        print("📝 Kết quả nhận diện văn bản:")
        print(decoded_text if decoded_text else "(Không nhận diện được văn bản)")

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
