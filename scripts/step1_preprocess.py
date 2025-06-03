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
    # Làm sạch câu: chữ thường, bỏ ký tự \n, \t
    df['sentence'] = df['sentence'].str.lower().str.replace(r"[\n\t]", "", regex=True)
    return df

# ===================== BUILD VOCAB & CHAR2IDX =====================
def build_vocab_mapping(sentences):
    # Lấy toàn bộ ký tự có trong tập câu
    vocab = sorted(set(''.join(sentences)))
    
    # Tạo dict char2idx với <blank> = 0 theo chuẩn CTC
    char2idx = {'<blank>': 0}
    for i, c in enumerate(vocab, start=1):
        char2idx[c] = i
    idx2char = {v: k for k, v in char2idx.items()}
    
    # Lưu vocab ra file để kiểm tra
    with open(os.path.join(PROCESSED_DIR, 'vocab.txt'), 'w', encoding='utf-8') as f:
        for char in vocab:
            f.write(f"{char}\n")
    
    print(f"[INFO] Vocab size (not include blank): {len(vocab)}")
    print(f"[INFO] Max index in char2idx: {max(char2idx.values())}")
    return char2idx, idx2char

# ===================== CHUYỂN TEXT SANG LABELS =====================
def text_to_labels(text, char2idx):
    labels = []
    unknown_chars = set()
    for ch in text:
        if ch in char2idx:
            labels.append(char2idx[ch])
        else:
            unknown_chars.add(ch)
    if unknown_chars:
        print(f"[WARNING] Ký tự lạ không có trong vocab: {unknown_chars}")
    return labels

# ===================== TIỀN XỬ LÝ ÂM THANH =====================
def preprocess_audio(full_audio_path):
    wav, _ = librosa.load(full_audio_path, sr=TARGET_SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(y=wav, sr=TARGET_SAMPLE_RATE, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.T.astype(np.float32)  # transpose về (time, n_mels)

# ===================== TIẾN TRÌNH XỬ LÝ & LƯU =====================
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

            # Lưu file spec và label riêng biệt
            np.save(os.path.join(PROCESSED_DIR, f'spec_{i}.npy'), spec)
            np.save(os.path.join(PROCESSED_DIR, f'label_{i}.npy'), label)

            if i % 500 == 0:
                print(f"[INFO] Đã xử lý và lưu {i + 1}/{len(df)} files")

        except Exception as e:
            print(f"[ERROR] File lỗi ({audio_path}): {e}")

    # Lưu char2idx ra file .npy để load dễ dàng sau này
    np.save(os.path.join(PROCESSED_DIR, 'char2idx.npy'), char2idx)
    print(f"[DONE] Hoàn tất xử lý và lưu dữ liệu tại '{PROCESSED_DIR}/'")
    
# ===================== KIỂM TRA TƯƠNG THÍCH LABEL/VOCAB =====================
def verify_labels_with_vocab(index_list, processed_dir, char2idx):
    max_idx = max(char2idx.values())
    print(f"[CHECK] Bắt đầu kiểm tra label và vocab...")

    for idx in index_list:
        label_path = os.path.join(processed_dir, f'label_{idx}.npy')
        if not os.path.exists(label_path):
            print(f"[WARNING] Không tìm thấy file: label_{idx}.npy")
            continue

        label = np.load(label_path)
        if np.any(label > max_idx):
            print(f"[ERROR] label_{idx}.npy chứa chỉ số vượt max ({max_idx}): {label}")
        if np.any(label < 1):  # nhỏ hơn 1 (vì 0 là blank, nhưng không nên có trong label)
            print(f"[ERROR] label_{idx}.npy chứa chỉ số không hợp lệ (<1): {label}")

    print(f"[DONE] Kiểm tra label/vocab hoàn tất.")

# ===================== MAIN =====================
if __name__ == "__main__":
    df = load_tsv_data(TSV_PATH)
    char2idx, idx2char = build_vocab_mapping(df['sentence'])

    print(f"[INFO] Bảng char2idx sample (first 20 items):")
    for i, (ch, idx) in enumerate(char2idx.items()):
        if i >= 20:
            break
        print(f"  '{ch}': {idx}")

    preprocess_and_save(df, CLIPS_DIR, char2idx)
    
    index_list = list(range(len(df)))
    verify_labels_with_vocab(index_list, PROCESSED_DIR, char2idx)
