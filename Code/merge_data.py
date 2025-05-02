import os
import pandas as pd
import shutil

# Đường dẫn đến thư mục chứa 15 folder dữ liệu
root_input_dir = r'D:\DATN\Data'
output_dir = 'combined_dataset'
output_clips_dir = os.path.join(output_dir, 'clips')
os.makedirs(output_clips_dir, exist_ok=True)

combined_train, combined_dev, combined_test = [], [], []

# Duyệt qua từng thư mục dữ liệu
for folder_id, subfolder in enumerate(sorted(os.listdir(root_input_dir))):
    folder_path = os.path.join(root_input_dir, subfolder)
    if not os.path.isdir(folder_path):
        continue

    print(f"🔍 Đang xử lý folder: {folder_path}")

    # Tìm thư mục chứa clips (thường là .../vi/)
    subdirs = [os.path.join(folder_path, d) for d in os.listdir(folder_path)
               if os.path.isdir(os.path.join(folder_path, d))]
    clips_dir, tsv_dir = None, None

    for d in subdirs:
        if os.path.exists(os.path.join(d, 'clips')):
            clips_dir = os.path.join(d, 'clips')
            tsv_dir = d
            break

    if clips_dir is None or tsv_dir is None:
        print(f"⚠️ Thiếu thư mục clips trong: {folder_path}")
        continue

    for split in ['train', 'dev', 'test']:
        tsv_file = os.path.join(tsv_dir, f"{split}.tsv")
        if not os.path.exists(tsv_file):
            print(f"⚠️ Thiếu file {split}.tsv trong: {tsv_dir}")
            continue

        df = pd.read_csv(tsv_file, sep='\t')
        print(f"✅ Đã đọc {len(df)} dòng từ {tsv_file}")

        for i, row in df.iterrows():
            old_filename = row['path']
            new_filename = f"clip_{folder_id}_{i}.mp3"
            old_file_path = os.path.join(clips_dir, old_filename)
            new_file_path = os.path.join(output_clips_dir, new_filename)

            if os.path.exists(old_file_path):
                shutil.copyfile(old_file_path, new_file_path)
                row['path'] = new_filename
                if split == 'train':
                    combined_train.append(row.to_dict())
                elif split == 'dev':
                    combined_dev.append(row.to_dict())
                elif split == 'test':
                    combined_test.append(row.to_dict())
            else:
                print(f"⚠️ Thiếu file âm thanh: {old_file_path}")

# Ghi dữ liệu hợp nhất
os.makedirs(output_dir, exist_ok=True)
pd.DataFrame(combined_train).to_csv(os.path.join(output_dir, 'combined_train.tsv'), sep='\t', index=False)
print(f"✅ Đã lưu train ({len(combined_train)} dòng) vào {output_dir}\\combined_train.tsv")

pd.DataFrame(combined_dev).to_csv(os.path.join(output_dir, 'combined_dev.tsv'), sep='\t', index=False)
print(f"✅ Đã lưu dev ({len(combined_dev)} dòng) vào {output_dir}\\combined_dev.tsv")

pd.DataFrame(combined_test).to_csv(os.path.join(output_dir, 'combined_test.tsv'), sep='\t', index=False)
print(f"✅ Đã lưu test ({len(combined_test)} dòng) vào {output_dir}\\combined_test.tsv")

print("🎉 Hoàn tất gộp dữ liệu từ 15 folder.")
