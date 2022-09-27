import os
import argparse
import torch
import torchaudio
from typing import List
import torch.nn.functional as F
import warnings
import string
import random

from glob import glob
from tqdm import tqdm

torchaudio.set_audio_backend("sox_io")


def random_str():
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(10))


def read_audio(path: str):
    wav, _ = torchaudio.load(path)

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    return wav.squeeze(0)


def save_audio(path: str, tensor: torch.Tensor, sampling_rate: int = 16000):
    torchaudio.save(path, tensor.unsqueeze(0), sampling_rate)


def init_jit_model(model_path: str, device=torch.device('cpu')):
    torch.set_grad_enabled(False)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


def get_speech_timestamps(audio: torch.Tensor,
                          model,
                          threshold: float = 0.5,
                          sampling_rate: int = 16000,
                          min_speech_duration_ms: int = 250,
                          min_silence_duration_ms: int = 100,
                          window_size_samples: int = 1536,
                          speech_pad_ms: int = 30,
                          return_seconds: bool = False):

    """
    This method is used for splitting long audios into speech chunks using silero VAD
    Parameters
    ----------
    audio: torch.Tensor, one dimensional
        One dimensional float torch.Tensor, other types are casted to torch if possible
    model: preloaded .jit silero VAD model
    threshold: float (default - 0.5)
        Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
        It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.
    sampling_rate: int (default - 16000)
        Currently silero VAD models support 8000 and 16000 sample rates
    min_speech_duration_ms: int (default - 250 milliseconds)
        Final speech chunks shorter min_speech_duration_ms are thrown out
    min_silence_duration_ms: int (default - 100 milliseconds)
        In the end of each speech chunk wait for min_silence_duration_ms before separating it
    window_size_samples: int (default - 1536 samples)
        Audio chunks of window_size_samples size are fed to the silero VAD model.
        WARNING! Silero VAD models were trained using 512, 1024, 1536 samples for 16000 sample rate and 256, 512, 768 samples for 8000 sample rate.
        Values other than these may affect model perfomance!!
    speech_pad_ms: int (default - 30 milliseconds)
        Final speech chunks are padded by speech_pad_ms each side
    return_seconds: bool (default - False)
        whether return timestamps in seconds (default - samples)
    Returns
    ----------
    speeches: list of dicts
        list containing ends and beginnings of speech chunks (samples or seconds based on return_seconds)
    """

    if not torch.is_tensor(audio):
        try:
            audio = torch.Tensor(audio)
        except:
            raise TypeError("Audio cannot be casted to tensor. Cast it manually")

    if len(audio.shape) > 1:
        for i in range(len(audio.shape)):  # trying to squeeze empty dimensions
            audio = audio.squeeze(0)
        if len(audio.shape) > 1:
            raise ValueError("More than one dimension in audio. Are you trying to process audio with 2 channels?")

    if sampling_rate > 16000 and (sampling_rate % 16000 == 0):
        step = sampling_rate // 16000
        sampling_rate = 16000
        audio = audio[::step]
        warnings.warn('Sampling rate is a multiply of 16000, casting to 16000 manually!')
    else:
        step = 1

    if sampling_rate == 8000 and window_size_samples > 768:
        warnings.warn('window_size_samples is too big for 8000 sampling_rate! Better set window_size_samples to 256, 512 or 768 for 8000 sample rate!')
    if window_size_samples not in [256, 512, 768, 1024, 1536]:
        warnings.warn('Unusual window_size_samples! Supported window_size_samples:\n - [512, 1024, 1536] for 16000 sampling_rate\n - [256, 512, 768] for 8000 sampling_rate')

    model.reset_states()
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000

    audio_length_samples = len(audio)

    speech_probs = []
    for current_start_sample in range(0, audio_length_samples, window_size_samples):
        chunk = audio[current_start_sample: current_start_sample + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = torch.nn.functional.pad(chunk, (0, int(window_size_samples - len(chunk))))
        speech_prob = model(chunk, sampling_rate).item()
        speech_probs.append(speech_prob)

    triggered = False
    speeches = []
    current_speech = {}
    neg_threshold = threshold - 0.15
    temp_end = 0

    for i, speech_prob in enumerate(speech_probs):
        if (speech_prob >= threshold) and temp_end:
            temp_end = 0

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech['start'] = window_size_samples * i
            continue

        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = window_size_samples * i
            if (window_size_samples * i) - temp_end < min_silence_samples:
                continue
            else:
                current_speech['end'] = temp_end
                if (current_speech['end'] - current_speech['start']) > min_speech_samples:
                    speeches.append(current_speech)
                temp_end = 0
                current_speech = {}
                triggered = False
                continue

    if current_speech and (audio_length_samples - current_speech['start']) > min_speech_samples:
        current_speech['end'] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech['start'] = int(max(0, speech['start'] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i+1]['start'] - speech['end']
            if silence_duration < 2 * speech_pad_samples:
                speech['end'] += int(silence_duration // 2)
                speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - silence_duration // 2))
            else:
                speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))
                speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - speech_pad_samples))
        else:
            speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))

    if return_seconds:
        for speech_dict in speeches:
            speech_dict['start'] = round(speech_dict['start'] / sampling_rate, 1)
            speech_dict['end'] = round(speech_dict['end'] / sampling_rate, 1)
    elif step > 1:
        for speech_dict in speeches:
            speech_dict['start'] *= step
            speech_dict['end'] *= step

    return speeches


def run(current_folder, args):
    # get the path to the "silero_vad.jit" file
    vad_path = args.silero_vad
    if '/' not in vad_path:
        vad_path = current_folder + '/' + args.silero_vad

    # fix the number of threads (as it is shown in the Silero demos)
    torch.set_num_threads(1)

    # init the model
    model = init_jit_model(vad_path)

    # scan files
    mask = f'*.{args.format}'
    if args.recursive:
        mask = f'**/*.{args.format}'

    # get all files to chunk
    path_mask = args.source + '/' + mask
    files = glob(path_mask)

    print(f'Searching files as: {path_mask}')
    print(f'Found files: {len(files)}')

    if len(files) > 0:
        for filename in tqdm(files):
            wav = read_audio(filename)
            speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000, window_size_samples=512)

            # set the chunk folder
            if args.random_subfolder:
                subfolder = f'{args.destination}/' + random_str()
                os.makedirs(subfolder, exist_ok=True)

                chunk_folder = subfolder
            else:
                chunk_folder = args.destination

            # save chunks
            for n, chunk in enumerate(speech_timestamps):
                # extract voice by "start" and "end"
                chunk_wav = wav[chunk['start'] : chunk['end']]

                original_filename = filename.split('/')[-1].replace('.wav', '')
                chunk_filename = f'{chunk_folder}/{original_filename}__chunk_{n}.wav'

                # if the file does not exist, save it
                if not os.path.exists(chunk_filename):
                    save_audio(chunk_filename, chunk_wav, 16000)

                # remove those chunks we don't need
                if args.min_chunk_len and args.max_chunk_len:
                    info = torchaudio.info(chunk_filename)
                    duration = info.num_frames / info.sample_rate
                    
                    if duration < args.min_chunk_len or duration > args.max_chunk_len:
                        os.remove(chunk_filename)

    print('Finished.')


def validate_args(current_folder, args):
    # check existence of source and destination folders
    if not os.path.exists(args.source):
        print(f'The folder {args.source} does not exist')
        exit(1)
    
    if not os.path.exists(args.destination):
        print(f'The folder {args.destination} does not exist')
        exit(1)

    # check existence of the "silero_vad.jit" file
    vad_path = args.silero_vad
    if '/' not in vad_path:
        vad_path = current_folder + '/' + args.silero_vad
    if not os.path.exists(vad_path):
        print(f'The file {vad_path} does not exist')
        exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="A tool to slice your audio files into chunks using Voice Activity Detection technique."
    )

    parser.add_argument(
        "--silero_vad",
        help="Path to a silero_vad.jit file",
        type=str,
        default="./silero_vad.jit",
    )
    parser.add_argument(
        "--source",
        help="Path to a folder with audio files",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--destination",
        help="Path to a folder where to save chunks",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--min_chunk_len",
        help="Minimal length of a chunk",
        type=float,
        required=False,
    )
    parser.add_argument(
        "--max_chunk_len",
        help="Maximal length of a chunk",
        type=float,
        required=False,
    )
    parser.add_argument(
        "--format",
        help="Format of audio files: WAV only yet",
        type=str,
        required=True,
    )
    parser.add_argument(
        '--recursive', 
        help="Search audio files recursively",
        action='store_true',
    )
    parser.add_argument(
        '--random_subfolder', 
        help="Save chunks in a subfolder generated randomly",
        action='store_true',
    )
    
    args = parser.parse_args()

    # get the current folder
    current_folder = os.getcwd()

    # first, validate arguments
    validate_args(current_folder, args)

    # run the app
    run(current_folder, args)


if __name__ == "__main__":
    main()
