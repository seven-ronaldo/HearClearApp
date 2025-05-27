import os
import pandas as pd
import shutil

# ===================== CẤU HÌNH =====================
root_input_dir = r'D:\DATN\Data'
output_dir = 'combined_dataset'
output_clips_dir = os.path.join(output_dir, 'clips')
os.makedirs(output_clips_dir, exist_ok=True)

combined_train, combined_dev, combined_test = [], [], []
global_clip_counter = 0  # Số thứ tự duy nhất cho mỗi file .mp3

# ===================== HÀM XỬ LÝ =====================
def process_split(split, df, clips_dir):
    global global_clip_counter
    combined_list = []

    for _, row in df.iterrows():
        old_filename = row['path']
        old_file_path = os.path.join(clips_dir, old_filename)

        if not os.path.exists(old_file_path):
            print(f"⚠️ Không tìm thấy file âm thanh: {old_file_path}")
            continue

        # Đặt tên file mới
        new_filename = f"clip_{global_clip_counter}.mp3"
        new_file_path = os.path.join(output_clips_dir, new_filename)

        try:
            shutil.copyfile(old_file_path, new_file_path)
            row['path'] = new_filename
            combined_list.append(row)
            global_clip_counter += 1
        except Exception as e:
            print(f"❌ Lỗi khi copy {old_file_path}: {e}")

    return combined_list

# ===================== XỬ LÝ TỪNG FOLDER =====================
for subfolder in sorted(os.listdir(root_input_dir)):
    folder_path = os.path.join(root_input_dir, subfolder)
    if not os.path.isdir(folder_path):
        continue

    print(f"\n🔍 Đang xử lý folder: {folder_path}")

    subdirs = [os.path.join(folder_path, d) for d in os.listdir(folder_path)
               if os.path.isdir(os.path.join(folder_path, d))]

    clips_dir, tsv_dir = None, None
    for d in subdirs:
        if os.path.exists(os.path.join(d, 'clips')):
            clips_dir = os.path.join(d, 'clips')
            tsv_dir = d
            break

    if clips_dir is None or tsv_dir is None:
        print(f"⚠️ Không tìm thấy thư mục clips trong {folder_path}")
        continue

    for split in ['train', 'dev', 'test']:
        tsv_file = os.path.join(tsv_dir, f"{split}.tsv")
        if not os.path.exists(tsv_file):
            print(f"⚠️ Thiếu file {split}.tsv trong: {tsv_dir}")
            continue

        df = pd.read_csv(tsv_file, sep='\t')
        print(f"✅ Đọc {len(df)} dòng từ {tsv_file}")

        combined = process_split(split, df, clips_dir)
        if split == 'train':
            combined_train.extend(combined)
        elif split == 'dev':
            combined_dev.extend(combined)
        elif split == 'test':
            combined_test.extend(combined)

# ===================== GHI DỮ LIỆU =====================
os.makedirs(output_dir, exist_ok=True)

pd.DataFrame(combined_train).to_csv(os.path.join(output_dir, 'combined_train.tsv'), sep='\t', index=False)
print(f"\n✅ Train: {len(combined_train)} dòng → {output_dir}\\combined_train.tsv")

pd.DataFrame(combined_dev).to_csv(os.path.join(output_dir, 'combined_dev.tsv'), sep='\t', index=False)
print(f"✅ Dev: {len(combined_dev)} dòng → {output_dir}\\combined_dev.tsv")

pd.DataFrame(combined_test).to_csv(os.path.join(output_dir, 'combined_test.tsv'), sep='\t', index=False)
print(f"✅ Test: {len(combined_test)} dòng → {output_dir}\\combined_test.tsv")

print(f"\n🎉 Tổng cộng {global_clip_counter} file .mp3 đã được xử lý.")
print(f"[CHECK] Số lượng file thật: {len(os.listdir(output_clips_dir))}")
