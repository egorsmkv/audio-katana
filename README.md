# audio-katana

A tool to slice your audio files into chunks using Voice Activity Detection technique

## Requirements

- Audio files must be in 16 kHz

## Install deps

```bash
pip install torch torchaudio tqdm
```

## Usage

Basic command:

```bash
python audio_katana.py --source ./demo --format wav --destination ./demo-chunks
```

Create own folder for chunks:

```bash
python audio_katana.py --random_subfolder --source ./demo --format wav --destination ./demo-chunks
```

Find audio files recursively:

```bash
python audio_katana.py --recursive --random_subfolder --source ./demo --format wav --destination ./demo-chunks
```