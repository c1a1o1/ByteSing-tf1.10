# coding:utf-
import argparse
import os
from multiprocessing import cpu_count
import wave
from pydub import AudioSegment
import music21 as m21
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from functools import partial
from tqdm import tqdm

from myData import pinyin
from datasets import audio
from hparams import hparams


def get_second_part_wave(wav, start_time, end_time, hparams):
    start_time = int(start_time * 1000)
    end_time = int(end_time * 1000)
    sentence = wav[start_time: end_time]
    temp = sentence.export('temp.wav', format="wav")
    sentence = audio.load_wav('temp.wav', sr=hparams.sample_rate)
    return sentence

def get_music_score(metadata_filename):
    # 处理乐谱，输出每个音素[持续时长，midi，因素类型，音素]
    lines = []
    score = m21.converter.parse(metadata_filename)
    part = score.parts.flat
    for i in range(len(part.notesAndRests)):
        event = part.notesAndRests[i]
        if isinstance(event, m21.note.Note):
            duration = event.seconds
            midi = event.pitch.midi
            if len(event.lyrics) > 0:
                token = event.lyrics[1].text+'3'
                token = pinyin.split_pinyin(token)
                if token[0] != '':              
                    lines.append([duration, midi, 0, token[0]])
                    lines.append([duration, midi, 1, token[1]])
                elif token[1] != '':
                    lines.append([duration, midi, 2, token[1]])
            else:
                temp = lines[-1]
                lines[-1][0] = lines[-1][0] + duration
        elif isinstance(event, m21.note.Rest):
            duration = event.seconds
            midi = 0
            token = 'sp'
            if lines[-1][-1] != 'sp':
                lines.append([duration, midi, 2, token])
            else:
                lines[-1][0] = lines[-1][0] + duration
    return lines

def get_phoneme_duration(metadata_filename):
    # 处理音频时长标注信息，返回[开始时间，结束时间，对应音素]
    with open(metadata_filename, encoding='utf-8') as f:
        i = 0
        j = 0
        durationOutput = []
        for line in f:
            if j != 15:
                j = j+1
                continue
            line = line.split('\n')[0]
            if i == 0:
                startTime = float(line)
                i = i+1
            elif i == 1:
                endTime = float(line)
                i = i+1
            else:
                i = 0
                temp = line.split('"')[1]
                if temp == 'sil' or temp == 'pau':
                    temp = 'sp'
                if j == 15:
                    durationOutput.append([startTime, endTime, temp])
                else:
                    if durationOutput[-1][2] != temp:
                        durationOutput.append([startTime, endTime, temp])
                    else:
                        durationOutput[-1][1] = endTime
    return durationOutput

def audio_process_utterance(mel_dir, linear_dir, wav_dir, duration_dir, score_dir, index, wav, durations, scores, hparams):
    """
    Preprocesses a single utterance wav/text pair

    this writes the mel scale spectogram to disk and return a tuple to write
    to the train.txt file

    Args:
    	- mel_dir: the directory to write the mel spectograms into
    	- linear_dir: the directory to write the linear spectrograms into
    	- wav_dir: the directory to write the preprocessed wav into
    	- index: the numeric index to use in the spectogram filename
    	- wav_path: path to the audio file containing the speech input
    	- hparams: hyper parameters

    Returns:
    	- A tuple: (audio_filename, mel_filename, linear_filename, score_filename, duration_filename, time_steps, mel_frames)
    """
    #rescale wav
    if hparams.rescale:
    	wav = wav / np.abs(wav).max() * hparams.rescaling_max

    #Get spectrogram from wav
    ret = audio.wav2spectrograms(wav, hparams)
    if ret is None:
    	return None
    out = ret[0]
    mel_spectrogram = ret[1]
    linear_spectrogram = ret[2]
    time_steps = ret[3]
    mel_frames = ret[4]

    # Write the spectrogram and audio to disk
    audio_filename = 'audio-{}.npy'.format(index)
    mel_filename = 'mel-{}.npy'.format(index)
    linear_filename = 'linear-{}.npy'.format(index)
    duration_filename = 'duration-{}.npy'.format(index)
    score_filename = 'score-{}.npy'.format(index)
    np.save(os.path.join(wav_dir, audio_filename), out.astype(np.float32), allow_pickle=False)
    np.save(os.path.join(mel_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
    np.save(os.path.join(linear_dir, linear_filename), linear_spectrogram.T, allow_pickle=False)    
    np.save(os.path.join(duration_dir, duration_filename), durations, allow_pickle=False)
    np.save(os.path.join(score_dir, score_filename), scores, allow_pickle=False)

    # Return a tuple describing this training example
    return (audio_filename, mel_filename, linear_filename, duration_filename, score_filename, time_steps, mel_frames)

def build_from_path(hparams, input_dir, mel_dir, linear_dir, wav_dir, score_dir, duration_dir, n_jobs=12, tqdm=lambda x: x):
    """
    Args:
    	- hparams: hyper parameters
    	- input_dir: input directory that contains the files to prerocess
    	- mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
    	- linear_dir: output directory of the preprocessed speech linear-spectrogram dataset
    	- wav_dir: output directory of the preprocessed speech audio dataset
    	- n_jobs: Optional, number of worker process to parallelize across
    	- tqdm: Optional, provides a nice progress bar

    Returns:
    	- A list of tuple describing the train examples. this should be written to train.txt
    """

    # We use ProcessPoolExecutor to parallelize across processes, this is just for
    # optimization purposes and it can be omited
    executor = ProcessPoolExecutor(max_workers=n_jobs)
    scores = get_music_score(os.path.join(input_dir, '001.musicxml'))
    durations = get_phoneme_duration(os.path.join(input_dir, '001.interval'))
    song = AudioSegment.from_wav(os.path.join(input_dir, '001.wav'))
    futures = []
    index = 1    
    sentence_duration = []
    score_index = -1
    for i in range(len(scores)):
        sentence_duration.append(durations[i])
        if durations[i][2] == 'sp':
            sentence_score = []
            wav = get_second_part_wave(song, sentence_duration[0][0], sentence_duration[-1][0], hparams)
            while True:
                score_index += 1
                sentence_score.append(scores[score_index])
                if scores[score_index][3] == 'sp':
                    futures.append(executor.submit(partial(audio_process_utterance, mel_dir, linear_dir, wav_dir,\
                           duration_dir, score_dir, index, wav, sentence_duration, sentence_score, hparams)))
                    # futures.append(audio_process_utterance(mel_dir, linear_dir, wav_dir,\
                    #        duration_dir, score_dir, index, wav, sentence_duration, sentence_score, hparams))
                    index += 1
                    sentence_duration = []
                    break

    return [future.result() for future in tqdm(futures) if future.result() is not None]
    # return futures

def write_metadata(metadata, out_dir):
	with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
		for m in metadata:
			f.write('|'.join([str(x) for x in m]) + '\n')
	mel_frames = sum([int(m[6]) for m in metadata])
	timesteps = sum([int(m[5]) for m in metadata])
	sr = hparams.sample_rate
	hours = timesteps / sr / 3600
	print('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(
		len(metadata), mel_frames, timesteps, hours))
	print('Max mel frames length: {}'.format(max(int(m[6]) for m in metadata)))
	print('Max audio timesteps length: {}'.format(max(m[5] for m in metadata)))

def main():
    print('initializing preprocessing..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='/datapool/home/ywy19/singing-synthesis/ByteSing')
    parser.add_argument('--hparams', default='',
    	help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--dataset', default='myData')
    parser.add_argument('--output', default='training_data')
    parser.add_argument('--n_jobs', type=int, default=cpu_count())
    args = parser.parse_args()

    modified_hp = hparams.parse(args.hparams)
	
	# Prepare directories
    in_dir  = os.path.join(args.base_dir, args.dataset)
    out_dir = os.path.join(args.base_dir, args.output)
    mel_dir = os.path.join(out_dir, 'mels')
    wav_dir = os.path.join(out_dir, 'audio')
    lin_dir = os.path.join(out_dir, 'linear')
    dur_dir = os.path.join(out_dir, 'duration')
    sco_dir = os.path.join(out_dir, 'score')
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(lin_dir, exist_ok=True)
    os.makedirs(dur_dir, exist_ok=True)
    os.makedirs(sco_dir, exist_ok=True)
	
	# Process dataset
    metadata = []
    metadata = build_from_path(modified_hp, in_dir, mel_dir, lin_dir, wav_dir, sco_dir, dur_dir, args.n_jobs, tqdm=tqdm)
	# Write metadata to 'train.txt' for training
    write_metadata(metadata, out_dir)


if __name__ == '__main__':
	main()

    

