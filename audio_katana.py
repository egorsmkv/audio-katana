import os
import argparse
import torch
import torchaudio
from typing import List
import torch.nn.functional as F
import warnings
import string
import random
import librosa
import numpy as np

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


def wada_snr(wav):
    # Direct blind estimation of the SNR of a speech signal.
    #
    # Paper on WADA SNR:
    #   http://www.cs.cmu.edu/~robust/Papers/KimSternIS08.pdf
    #
    # This function was adapted from this matlab code:
    #   https://labrosa.ee.columbia.edu/projects/snreval/#9

    # init
    eps = 1e-10
    # next 2 lines define a fancy curve derived from a gamma distribution -- see paper
    db_vals = np.arange(-20, 101)
    g_vals = np.array([0.40974774, 0.40986926, 0.40998566, 0.40969089, 0.40986186, 0.40999006, 0.41027138, 0.41052627, 0.41101024, 0.41143264, 0.41231718, 0.41337272, 0.41526426, 0.4178192 , 0.42077252, 0.42452799, 0.42918886, 0.43510373, 0.44234195, 0.45161485, 0.46221153, 0.47491647, 0.48883809, 0.50509236, 0.52353709, 0.54372088, 0.56532427, 0.58847532, 0.61346212, 0.63954496, 0.66750818, 0.69583724, 0.72454762, 0.75414799, 0.78323148, 0.81240985, 0.84219775, 0.87166406, 0.90030504, 0.92880418, 0.95655449, 0.9835349 , 1.01047155, 1.0362095 , 1.06136425, 1.08579312, 1.1094819 , 1.13277995, 1.15472826, 1.17627308, 1.19703503, 1.21671694, 1.23535898, 1.25364313, 1.27103891, 1.28718029, 1.30302865, 1.31839527, 1.33294817, 1.34700935, 1.3605727 , 1.37345513, 1.38577122, 1.39733504, 1.40856397, 1.41959619, 1.42983624, 1.43958467, 1.44902176, 1.45804831, 1.46669568, 1.47486938, 1.48269965, 1.49034339, 1.49748214, 1.50435106, 1.51076426, 1.51698915, 1.5229097 , 1.528578  , 1.53389835, 1.5391211 , 1.5439065 , 1.54858517, 1.55310776, 1.55744391, 1.56164927, 1.56566348, 1.56938671, 1.57307767, 1.57654764, 1.57980083, 1.58304129, 1.58602496, 1.58880681, 1.59162477, 1.5941969 , 1.59693155, 1.599446  , 1.60185011, 1.60408668, 1.60627134, 1.60826199, 1.61004547, 1.61192472, 1.61369656, 1.61534074, 1.61688905, 1.61838916, 1.61985374, 1.62135878, 1.62268119, 1.62390423, 1.62513143, 1.62632463, 1.6274027 , 1.62842767, 1.62945532, 1.6303307 , 1.63128026, 1.63204102])

    # peak normalize, get magnitude, clip lower bound
    wav = np.array(wav)
    wav = wav / abs(wav).max()
    abs_wav = abs(wav)
    abs_wav[abs_wav < eps] = eps

    # calcuate statistics
    # E[|z|]
    v1 = max(eps, abs_wav.mean())
    # E[log|z|]
    v2 = np.log(abs_wav).mean()
    # log(E[|z|]) - E[log(|z|)]
    v3 = np.log(v1) - v2

    # table interpolation
    wav_snr_idx = None
    if any(g_vals < v3):
        wav_snr_idx = np.where(g_vals < v3)[0].max()
    # handle edge cases or interpolate
    if wav_snr_idx is None:
        wav_snr = db_vals[0]
    elif wav_snr_idx == len(db_vals) - 1:
        wav_snr = db_vals[-1]
    else:
        wav_snr = db_vals[wav_snr_idx] + \
            (v3-g_vals[wav_snr_idx]) / (g_vals[wav_snr_idx+1] - \
            g_vals[wav_snr_idx]) * (db_vals[wav_snr_idx+1] - db_vals[wav_snr_idx])

    # Calculate SNR
    dEng = sum(wav**2)
    dFactor = 10**(wav_snr / 10)
    dNoiseEng = dEng / (1 + dFactor) # Noise energy
    dSigEng = dEng * dFactor / (1 + dFactor) # Signal energy
    snr = 10 * np.log10(dSigEng / dNoiseEng)

    return snr


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

                # filter out chunks by WADA SNR
                if args.min_wada_snr and args.max_wada_snr:
                    wav_data, _ = librosa.load(chunk_filename)
                    wada_snr_sig = wada_snr(wav_data)

                    if wada_snr_sig < args.min_wada_snr or wada_snr_sig > args.max_wada_snr:
                        os.remove(chunk_filename)

                # filter out chunks by length
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
        "--min_wada_snr",
        help="Minimal WADA SNR value",
        type=float,
        required=False,
    )
    parser.add_argument(
        "--max_wada_snr",
        help="Maximal WADA SNR value",
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
