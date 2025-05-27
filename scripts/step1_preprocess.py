# Load .tsv, audio → spectrogram + labels

import os
import librosa
import numpy as np
import pandas as pd

# ===================== CẤU HÌNH =====================
TSV_PATH = 'combined_dataset/combined_train.tsv'
CLIPS_DIR = 'combined_dataset/converted_wav'
PROCESSED_DIR = 'processed_chunks'
TARGET_SAMPLE_RATE = 16000
N_MELS = 128

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ===================== LOAD TSV =====================
def load_tsv_data(tsv_path):
    df = pd.read_csv(tsv_path, sep='\t')[['path', 'sentence']].dropna()
    df['sentence'] = df['sentence'].str.lower().str.replace(r"[\n\t]", "", regex=True)
    return df

# ===================== CHAR MAPPING =====================
def build_vocab_mapping(sentences):
    vocab = sorted(set(''.join(sentences)))
    char2idx = {c: i + 1 for i, c in enumerate(vocab)}  # start from 1
    char2idx['<blank>'] = 0  # for CTC
    idx2char = {v: k for k, v in char2idx.items()}
    
    # Save vocab to file
    with open(os.path.join(PROCESSED_DIR, 'vocab.txt'), 'w', encoding='utf-8') as f:
        for char in vocab:
            f.write(f"{char}\n")
    
    print(f"[INFO] Vocab size (not include blank): {len(vocab)}")
    print(f"[INFO] Max index: {max(char2idx.values())}")
    return char2idx, idx2char

def text_to_labels(text, char2idx):
    labels = []
    for ch in text:
        if ch in char2idx:
            idx = char2idx[ch]
            if idx >= len(char2idx):
                print(f"[ERROR] Index out of range for '{ch}': {idx}")
            labels.append(idx)
        else:
            print(f"[WARNING] Ký tự không có trong vocab: '{ch}'")
    return labels

# ===================== AUDIO PREPROCESS =====================
def preprocess_audio(full_audio_path):
    wav, _ = librosa.load(full_audio_path, sr=TARGET_SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(y=wav, sr=TARGET_SAMPLE_RATE, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.T.astype(np.float32)

# ===================== TIẾN TRÌNH XỬ LÝ =====================
def preprocess_and_save(df, clips_dir, char2idx):
    for i, row in df.iterrows():
        audio_path = os.path.join(clips_dir, row['path'])
        text = row['sentence']

        if not os.path.isfile(audio_path):
            print(f"[WARNING] File không tồn tại: {audio_path}")
            continue

        try:
            spec = preprocess_audio(audio_path)
            label = np.array(text_to_labels(text, char2idx), dtype=np.int32)

            # Lưu riêng từng file
            np.save(os.path.join(PROCESSED_DIR, f'spec_{i}.npy'), spec)
            np.save(os.path.join(PROCESSED_DIR, f'label_{i}.npy'), label)

            if i % 500 == 0:
                print(f"[INFO] Đã xử lý và lưu {i + 1}/{len(df)} files")

        except Exception as e:
            print(f"[ERROR] File lỗi ({audio_path}): {e}")

    # Lưu char2idx
    np.save(os.path.join(PROCESSED_DIR, 'char2idx.npy'), char2idx)
    print(f"[DONE] Hoàn tất xử lý và lưu dữ liệu tại '{PROCESSED_DIR}/'")

# ===================== MAIN =====================
if __name__ == "__main__":
    df = load_tsv_data(TSV_PATH)
    char2idx, idx2char = build_vocab_mapping(df['sentence'])
    preprocess_and_save(df, CLIPS_DIR, char2idx)

