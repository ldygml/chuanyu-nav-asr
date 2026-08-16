import os, glob

data_root = "/Save/Sichuan Dialect Scripted Speech Corpus"
wav_scp = "/root/asr-competition/official_data/wav.scp"
text_file = "/root/asr-competition/official_data/text"

os.makedirs("/root/asr-competition/official_data", exist_ok=True)

wav_count = 0
text_count = 0

with open(wav_scp, "w") as f_scp, open(text_file, "w") as f_txt:
    for city in sorted(os.listdir(data_root)):
        city_path = os.path.join(data_root, city)
        if not os.path.isdir(city_path):
            continue

        # Read transcription
        utter_file = os.path.join(city_path, "UTTERANCEINFO.txt")
        if not os.path.exists(utter_file):
            continue

        utt2text = {}
        with open(utter_file, "r") as f:
            header = f.readline()  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 6:
                    utt_id = parts[1].replace(".wav", "")  # G0003_S0001
                    text = parts[5]  # TRANSCRIPTION
                    if text:
                        utt2text[utt_id] = text

        # Find WAV files
        wav_dir = os.path.join(city_path, "WAV")
        if not os.path.exists(wav_dir):
            continue

        for spk_dir in sorted(os.listdir(wav_dir)):
            spk_path = os.path.join(wav_dir, spk_dir)
            if not os.path.isdir(spk_path):
                continue
            for wav_file in sorted(os.listdir(spk_path)):
                if not wav_file.endswith(".wav"):
                    continue
                utt_id = wav_file.replace(".wav", "")
                wav_path = os.path.join(spk_path, wav_file)

                f_scp.write(f"{utt_id}\t{wav_path}\n")
                wav_count += 1

                if utt_id in utt2text:
                    f_txt.write(f"{utt_id}\t{utt2text[utt_id]}\n")
                    text_count += 1

print(f"wav.scp: {wav_count} lines")
print(f"text: {text_count} lines")
