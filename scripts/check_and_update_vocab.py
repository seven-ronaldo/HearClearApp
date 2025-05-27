import os
import numpy as np

PROCESSED_DIR = "processed_chunks"

def check_and_update_vocab(processed_dir):
    # Load vocab hiện tại
    char2idx_path = os.path.join(processed_dir, 'char2idx.npy')
    if not os.path.exists(char2idx_path):
        raise FileNotFoundError(f"[ERROR] Không tìm thấy {char2idx_path}")
    
    char2idx = np.load(char2idx_path, allow_pickle=True).item()
    idx2char = {v: k for k, v in char2idx.items()}
    vocab_chars = set(idx2char.values())
    next_index = max(char2idx.values()) + 1

    unknown_chars = set()

    # Duyệt tất cả file label
    label_files = [f for f in os.listdir(processed_dir) if f.startswith('label_') and f.endswith('.npy')]
    print(f"🔍 Đang kiểm tra {len(label_files)} file label...")

    for f in label_files:
        try:
            label = np.load(os.path.join(processed_dir, f))
            chars = [idx2char.get(i, '[UNK]') for i in label]
            for c in chars:
                if c not in vocab_chars:
                    unknown_chars.add(c)
        except Exception as e:
            print(f"[LỖI] {f}: {e}")
            continue

    if not unknown_chars:
        print("✅ Vocab đã đầy đủ. Không cần cập nhật.")
        return

    # In ký tự mới
    print(f"⚠️ Phát hiện {len(unknown_chars)} ký tự mới ngoài vocab:")
    print(" ->", ", ".join(sorted(unknown_chars)))

    # Xác nhận tự động thêm
    print("➕ Đang cập nhật vocab...")
    for ch in sorted(unknown_chars):
        if ch not in char2idx:
            char2idx[ch] = next_index
            next_index += 1

    # Lưu lại
    np.save(char2idx_path, char2idx)
    print(f"✅ Đã cập nhật và lưu lại char2idx.npy với tổng {len(char2idx)} ký tự.")

if __name__ == "__main__":
    check_and_update_vocab(PROCESSED_DIR)
