import os
import pandas as pd
import librosa
import numpy as np
import tensorflow as tf

# ====================== STEP 1: LOAD DATA ======================
tsv_path = 'combined_dataset/combined_train.tsv'
clips_dir = 'combined_dataset/clips'

df = pd.read_csv(tsv_path, sep='\t')
df = df[['path', 'sentence']].dropna()

# ====================== STEP 2: CHAR MAPPING ======================
# Tạo từ điển chữ -> số
vocab = sorted(set(''.join(df['sentence'].str.lower())))
char2idx = {c: i + 1 for i, c in enumerate(vocab)}  # bắt đầu từ 1
char2idx['<blank>'] = 0  # cho CTC
idx2char = {v: k for k, v in char2idx.items()}

def text_to_labels(text):
    return [char2idx[c] for c in text.lower() if c in char2idx]

# ====================== STEP 3: AUDIO PROCESSING ======================
def preprocess_audio(path, target_sample_rate=16000):
    wav, _ = librosa.load(path, sr=target_sample_rate)
    mel = librosa.feature.melspectrogram(y=wav, sr=target_sample_rate, n_mels=128)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.T  # shape: (time, features)

# ====================== STEP 4: BUILD TF DATASET ======================
def generator():
    for _, row in df.iterrows():
        audio_path = os.path.join(clips_dir, row['path'])
        try:
            features = preprocess_audio(audio_path)
            labels = text_to_labels(row['sentence'])
            yield features, labels
        except Exception as e:
            print(f"Lỗi khi xử lý {audio_path}: {e}")
            continue

def get_dataset():
    output_types = (tf.float32, tf.int32)
    output_shapes = ([None, 128], [None])
    ds = tf.data.Dataset.from_generator(generator, output_types=output_types, output_shapes=output_shapes)
    ds = ds.padded_batch(32, padded_shapes=([None, 128], [None]))
    return ds

train_dataset = get_dataset()
