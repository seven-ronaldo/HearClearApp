import os
import pandas as pd

# Đường dẫn tới folder chứa các file TSV
tsv_dir = 'combined_dataset'
tsv_files = ['combined_train.tsv', 'combined_dev.tsv', 'combined_test.tsv']

for tsv_file in tsv_files:
    file_path = os.path.join(tsv_dir, tsv_file)
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy: {tsv_file}")
        continue

    df = pd.read_csv(file_path, sep='\t')
    if 'path' not in df.columns:
        print(f"⚠️ Cột 'path' không tồn tại trong {tsv_file}")
        continue

    df['path'] = df['path'].apply(lambda x: x.replace('.mp3', '.wav'))
    df.to_csv(file_path, sep='\t', index=False)
    print(f"✅ Đã cập nhật đường dẫn trong {tsv_file}")
