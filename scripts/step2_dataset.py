# Tạo tf.data.Dataset chuẩn

import os
import numpy as np
import tensorflow as tf

# ===================== CẤU HÌNH =====================
PROCESSED_DIR = 'processed_chunks'
BATCH_SIZE = 32
BUFFER_SIZE = 1000
N_MELS = 128  # số đặc trưng tần số (trục y)

# ===================== GENERATOR =====================
def data_generator_from_disk(index_list, processed_dir=PROCESSED_DIR):
    for idx in index_list:
        try:
            spec = np.load(os.path.join(processed_dir, f'spec_{idx}.npy'))
            label = np.load(os.path.join(processed_dir, f'label_{idx}.npy'))

            yield (
                spec.astype(np.float32),
                label.astype(np.int32),
                np.int32(spec.shape[0]),    # input_length (time steps)
                np.int32(label.shape[0])    # label_length (number of characters)
            )
        except Exception as e:
            print(f"[ERROR] Không load được index {idx}: {e}")
            continue

# ===================== TẠO DATASET THEO INDEX_LIST =====================
def create_dataset_from_index_list(index_list, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE):
    if not index_list:
        raise ValueError("[ERROR] index_list rỗng, không thể tạo dataset.")

    output_signature = (
        tf.TensorSpec(shape=(None, N_MELS), dtype=tf.float32),  # (time, n_mels)
        tf.TensorSpec(shape=(None,), dtype=tf.int32),           # (label_len,)
        tf.TensorSpec(shape=(), dtype=tf.int32),                # input_length
        tf.TensorSpec(shape=(), dtype=tf.int32),                # label_length
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator_from_disk(index_list),
        output_signature=output_signature
    )

    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.repeat()  # Lặp dataset vô hạn cho training
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=(
            [None, N_MELS],  # spectrogram
            [None],          # labels
            [],              # input_length
            []               # label_length
        ),
        padding_values=(
            0.0,  # spectrogram
            0,    # label
            0,    # input_length
            0     # label_length
        ),
        drop_remainder=True
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# ===================== LẤY DANH SÁCH INDEX =====================
def get_data_index_list(processed_dir=PROCESSED_DIR):
    index_list = sorted([
        int(f.split('_')[1].split('.')[0])
        for f in os.listdir(processed_dir)
        if f.startswith('spec_') and f.endswith('.npy')
    ])
    return index_list

# ===================== LOAD char2idx =====================
def load_char2idx(processed_dir=PROCESSED_DIR):
    path = os.path.join(processed_dir, 'char2idx.npy')
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Không tìm thấy file {path}")
    return np.load(path, allow_pickle=True).item()

# ===================== TEST =====================
if __name__ == "__main__":
    index_list = get_data_index_list()
    total_samples = len(index_list)
    train_idx = index_list[:int(0.9 * total_samples)]
    val_idx = index_list[int(0.9 * total_samples):]

    train_dataset = create_dataset_from_index_list(train_idx)
    val_dataset = create_dataset_from_index_list(val_idx)

    for batch in train_dataset.take(1):
        features, labels, input_len, label_len = batch
        print(f"Train batch features shape: {features.shape}")
        print(f"Train batch labels shape  : {labels.shape}")

    for batch in val_dataset.take(1):
        features, labels, input_len, label_len = batch
        print(f"Val batch features shape: {features.shape}")
        print(f"Val batch labels shape  : {labels.shape}")

    vocab = load_char2idx()
    print(f"Vocabulary size: {len(vocab)}")