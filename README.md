# Tacotron-2:
Tensorflow implementation of DeepMind's Tacotron-2. A deep neural network architecture described in this paper: [Natural TTS synthesis by conditioning Wavenet on MEL spectogram predictions](https://arxiv.org/pdf/1712.05884.pdf)

Forked from https://github.com/Rayhane-mamah/Tacotron-2 with [the commit on 2018.10.07] (https://github.com/Rayhane-mamah/Tacotron-2/tree/970b0803bb41e68cbac854dc958dbb03f34f9604)

Keeping only Griffin-Lim vocoder (WaveNet vocoder deleted).

# Repository Structure:
	Tacotron-2
	├── datasets
	├── LJSpeech-1.1	(0)
	│   └── wavs
	├── logs-Tacotron	(2)
	│   ├── eval_-dir
	│   │   ├── plots
	│   │   └── wavs
	│   ├── mel-spectrograms
	│   ├── plots
	│   ├── pretrained
	│   └── wavs
	├── tacotron
	│   ├── models
	│   └── utils
	├── tacotron_output	(3)
	│   ├── eval
	│   ├── gta
	│   ├── logs-eval
	│   │   ├── plots
	│   │   └── wavs
	│   └── natural
	└── training_data	(1)
	    ├── audio
	    ├── linear
	    └── mels


The previous tree shows the current state of the repository (separate training, one step at a time).

- Step **(0)**: Get your dataset, here I have set the examples of **Ljspeech**.
- Step **(1)**: Preprocess your data. This will give you the **training_data** folder.
- Step **(2)**: Train your Tacotron model. Yields the **logs-Tacotron** folder.
- Step **(3)**: Synthesize/Evaluate the Tacotron model. Gives the **tacotron_output** folder.


Note:
- **Our preprocessing only supports Ljspeech and Ljspeech-like datasets (M-AILABS speech data)!** If running on datasets stored differently, you will probably need to make your own preprocessing script.
- In the previous tree, files **were not represented** and **max depth was set to 3** for simplicity.
- If you run training of both **models at the same time**, repository structure will be different.

# Model Architecture:
<p align="center">
  <img src="https://preview.ibb.co/bU8sLS/Tacotron_2_Architecture.png"/>
</p>

The model described by the authors can be divided in two parts:
- Spectrogram prediction network
- Wavenet vocoder

To have an in-depth exploration of the model architecture, training procedure and preprocessing logic, refer to [our wiki](https://github.com/Rayhane-mamah/Tacotron-2/wiki)

# How to start
first, you need to have python 3 installed along with [Tensorflow](https://www.tensorflow.org/install/).

next you can install the requirements. If you are an Anaconda user: (else replace **pip** with **pip3** and **python** with **python3**)

> pip install -r requirements.txt

# Dataset:
We tested the code above on the [ljspeech dataset](https://keithito.com/LJ-Speech-Dataset/), which has almost 24 hours of labeled single actress voice recording. (further info on the dataset are available in the README file when you download it)

We are also running current tests on the [new M-AILABS speech dataset](http://www.m-ailabs.bayern/en/the-mailabs-speech-dataset/) which contains more than 700h of speech (more than 80 Gb of data) for more than 10 languages.

After **downloading** the dataset, **extract** the compressed file, and **place the folder inside the cloned repository.**

# Hparams setting:
Before proceeding, you must pick the hyperparameters that suit best your needs. While it is possible to change the hyper parameters from command line during preprocessing/training, I still recommend making the changes once and for all on the **hparams.py** file directly.

To pick optimal fft parameters, I have made a **griffin_lim_synthesis_tool** notebook that you can use to invert real extracted mel/linear spectrograms and choose how good your preprocessing is. All other options are well explained in the **hparams.py** and have meaningful names so that you can try multiple things with them.

# Preprocessing
Before running the following steps, please make sure you are inside **Tacotron-2 folder**

> cd Tacotron-2

Preprocessing can then be started using: 

> python preprocess.py

dataset can be chosen using the **--dataset** argument. If using M-AILABS dataset, you need to provide the **language, voice, reader, merge_books and book arguments** for your custom need. Default is **Ljspeech**.

This should take no longer than a **few minutes.**

# Training:
To **train the Tacotron-2 model** using:

> python train.py

checkpoints will be made each **5000 steps** and stored under **logs-Tacotron folder.**

**Note:**
- Please refer to train arguments under [train.py](https://github.com/Rayhane-mamah/Tacotron-2/blob/master/train.py) for a set of options you can use.

# Synthesis
To **synthesize audio** using:

> python synthesize.py

**Note:**
- Please refer to synthesis arguments under [synthesize.py](https://github.com/Rayhane-mamah/Tacotron-2/blob/master/synthesize.py) for a set of options you can use.


# References and Resources:
- [Original tacotron paper](https://arxiv.org/pdf/1703.10135.pdf)
- [Attention-Based Models for Speech Recognition](https://arxiv.org/pdf/1506.07503.pdf)
- [keithito/tacotron](https://github.com/keithito/tacotron)
- [Natural TTS synthesis by conditioning Wavenet on MEL spectogram predictions](https://arxiv.org/pdf/1712.05884.pdf)
- [Wavenet: A generative model for raw audio](https://arxiv.org/pdf/1609.03499.pdf)
- [Fast Wavenet](https://arxiv.org/pdf/1611.09482.pdf)
- [r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)

