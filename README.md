# audio-katana

A tool to slice your audio files into chunks using the Voice Activity Detection technique

## Install the project

```bash
git clone https://github.com/egorsmkv/audio-katana
cd audio-katana
```

## Install deps

```bash
pip install librosa torch torchaudio tqdm
```

## Usage

Basic command:

```bash
mkdir demo-chunks

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

Chunk audios and skip chunks out of limits: `--min_chunk_len`, `--max_chunk_len`:

```bash
python audio_katana.py --random_subfolder --source ./demo --format wav --destination ./demo-chunks --min_chunk_len 1.5 --max_chunk_len 5.5
```

Chunk audios and skip chunks out of WADA SNR values: `--min_wada_snr`, `--max_wada_snr`:

```bash
python audio_katana.py --random_subfolder --source ./demo --format wav --destination ./demo-chunks --min_wada_snr 1.5 --max_wada_snr 5.5
```
