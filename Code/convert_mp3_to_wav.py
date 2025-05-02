import os
from pydub import AudioSegment

# Đường dẫn thư mục chứa các file .mp3
input_dir = 'combined_dataset/clips'
output_dir = 'converted_wav'
os.makedirs(output_dir, exist_ok=True)

# Lặp qua tất cả các file mp3 trong thư mục
for filename in os.listdir(input_dir):
    if filename.endswith('.mp3'):
        mp3_path = os.path.join(input_dir, filename)
        wav_filename = filename.replace('.mp3', '.wav')
        wav_path = os.path.join(output_dir, wav_filename)

        try:
            sound = AudioSegment.from_mp3(mp3_path)
            sound = sound.set_channels(1).set_frame_rate(16000)
            sound.export(wav_path, format="wav")
            print(f"✅ Chuyển {filename} → {wav_filename}")
        except Exception as e:
            print(f"❌ Lỗi với {filename}: {e}")
