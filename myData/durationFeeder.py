import os
import threading
import time
import traceback

import numpy as np
import tensorflow as tf
from infolog import log
from sklearn.model_selection import train_test_split
import tacotron.utils.pinyin as py
from tacotron.utils.symbols import duration_symbols

_batches_per_group = 64

class Feeder:
	"""
		Feeds batches of data into queue on a background thread.
	"""

	def __init__(self, coordinator, metadata_filename, hparams):
		super(Feeder, self).__init__()
		self._coord = coordinator
		self._hparams = hparams
		self._cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
		self._train_offset = 0
		self._test_offset = 0

		# Load metadata
        self._dur_dir = os.path.join(os.path.dirname(metadata_filename), 'duration')
		self._score_dir = os.path.join(os.path.dirname(metadata_filename), 'score')
		# with open('./training_data/train.txt', encoding='utf-8') as f:
        with open(metadata_filename, encoding='utf-8') as f:
            self._metadatas = [line.strip().split('|') for line in f]
            log('Loaded metadata for {} examples').format(len(self._metadatas))

		#Train test split
		if hparams.duration_test_size is None:
			assert hparams.duration_test_batches is not None

		test_size = (hparams.duration_test_size if hparams.duration_test_size is not None
			else hparams.duration_test_batches * hparams.duration_batch_size)
		indices = np.arange(len(self._metadata))
		train_indices, test_indices = train_test_split(indices,
			test_size=test_size, random_state=hparams.tacotron_data_random_state)

		#Make sure test_indices is a multiple of batch_size else round up
		len_test_indices = self._round_down(len(test_indices), hparams.duration_batch_size)
		extra_test = test_indices[len_test_indices:]
		test_indices = test_indices[:len_test_indices]
		train_indices = np.concatenate([train_indices, extra_test])

		self._train_meta = list(np.array(self._metadata)[train_indices])
		self._test_meta = list(np.array(self._metadata)[test_indices])

		self.test_steps = len(self._test_meta) // hparams.duration_batch_size

		if hparams.duration_test_size is None:
			assert hparams.duration_test_batches == self.test_steps

		#pad input sequences with the <pad_token> 0 ( _ )
		self._pad = 0
		#explicitely setting the padding to a value that doesn't originally exist in the spectogram
		#to avoid any possible conflicts, without affecting the output range of the model too much
		if hparams.symmetric_mels:
			self._target_pad = -hparams.max_abs_value
		else:
			self._target_pad = 0.
		#Mark finished sequences with 1s
		self._token_pad = 1.

		with tf.device('/cpu:0'):
			# Create placeholders for inputs and targets. Don't specify batch size because we want
			# to be able to feed different batch sizes at eval time.
			self._placeholders = [
            # input: [phonemeID, phonemeType, duration]
			tf.placeholder(tf.int32, shape=(None, None), name='inputs_phoneme'),
			tf.placeholder(tf.int32, shape=(None, None), name='inputs_type'),
			tf.placeholder(tf.float32, shape=(None, None), name='inputs_time'),
			tf.placeholder(tf.int32, shape=(None, ), name='input_lengths'),
            # output: [phonemeID, startTime, endTime]
			tf.placeholder(tf.float32, shape=(None, None, 3), name='duration_targets'),
			tf.placeholder(tf.int32, shape=(None, ), name='targets_lengths'),
			tf.placeholder(tf.int32, shape=(hparams.tacotron_num_gpus, None), name='split_infos'),
			]

			# Create queue for buffering data
			queue = tf.FIFOQueue(8, [tf.int32, tf.int32, tf.float32, tf.int32, tf.float32, tf.int32, tf.int32], name='input_queue')
			self._enqueue_op = queue.enqueue(self._placeholders)
			self.inputs_phoneme, self.inputs_type, self.inputs_time, self.input_lengths, self.duration_targets, self.targets_lengths, self.split_infos = queue.dequeue()

			self.inputs_phoneme.set_shape(self._placeholders[0].shape)
			self.inputs_type.set_shape(self._placeholders[1].shape)
			self.inputs_time.set_shape(self._placeholders[2].shape)
			self.input_lengths.set_shape(self._placeholders[3].shape)			
			self.duration_targets.set_shape(self._placeholders[4].shape)
			self.targets_lengths.set_shape(self._placeholders[5].shape)
			self.split_infos.set_shape(self._placeholders[6].shape)

			# Create eval queue for buffering eval data
			eval_queue = tf.FIFOQueue(1, [tf.int32, tf.int32, tf.float32, tf.int32, tf.float32, tf.int32, tf.int32], name='eval_queue')
			self._eval_enqueue_op = eval_queue.enqueue(self._placeholders)
			self.eval_inputs_phoneme, self.eval_inputs_type, self.eval_inputs_time, self.eval_input_lengths, \
				self.eval_duration_targets, self.eval_targets_lengths, self.eval_split_infos = eval_queue.dequeue()

			self.eval_inputs_phoneme.set_shape(self._placeholders[0].shape)
			self.eval_inputs_type.set_shape(self._placeholders[1].shape)
			self.eval_inputs_time.set_shape(self._placeholders[2].shape)
			self.eval_input_lengths.set_shape(self._placeholders[3].shape)
			self.eval_duration_targets.set_shape(self._placeholders[4].shape)
			self.eval_targets_lengths.set_shape(self._placeholders[5].shape)
			self.eval_split_infos.set_shape(self._placeholders[6].shape)

	def start_threads(self, session):
		self._session = session
		thread = threading.Thread(name='background', target=self._enqueue_next_train_group)
		thread.daemon = True #Thread will close when parent quits
		thread.start()

		thread = threading.Thread(name='background', target=self._enqueue_next_test_group)
		thread.daemon = True #Thread will close when parent quits
		thread.start()

	def _get_test_groups(self):
		meta = self._test_meta[self._test_offset]
		self._test_offset += 1
		
		score_data = np.load(os.path.join(self._score_dir, meta[4]))
		# score.npy中的存储格式为：['音素所属音符理论时长'，'音符midi', '音素类型', '音素']
		inputs_phoneme = []
		inputs_type = []
		inputs_time = []
		for item in score_data:
			inputs_phoneme.append(self._onehot_encoding(item[3], duration_symbols[0]))
			inputs_type.append(self._onehot_encoding(item[2], duration_symbols[2]))
			inputs_time.append(float(item[0]))
		
		# duration.npy中的存储格式为：['开始时间', '结束时间', '音素']
		dur_data = np.load(os.path.join(self._dur_dir, meta[3]))
		dur_target = [self._dur_to_target(item) for item in dur_data]
		return (inputs_phoneme, inputs_type, inputs_time, dur_target, len(dur_target))

	def make_test_batches(self):
		start = time.time()

		# Read a group of examples
		n = self._hparams.duration_batch_size
		r = self._hparams.outputs_per_step

		#Test on entire test set
		examples = [self._get_test_groups() for i in range(len(self._test_meta))]

		# Bucket examples based on similar output sequence length for efficiency
		examples.sort(key=lambda x: x[-1])
		batches = [examples[i: i+n] for i in range(0, len(examples), n)]
		np.random.shuffle(batches)

		log('\nGenerated {} test batches of size {} in {:.3f} sec'.format(len(batches), n, time.time() - start))
		return batches, r

	def _enqueue_next_train_group(self):
		while not self._coord.should_stop():
			start = time.time()

			# Read a group of examples
			n = self._hparams.duration_batch_size
			r = self._hparams.outputs_per_step
			examples = [self._get_next_example() for i in range(n * _batches_per_group)]

			# Bucket examples based on similar output sequence length for efficiency
			examples.sort(key=lambda x: x[-1]) # 以dur_length为标准对examples排序
			batches = [examples[i: i+n] for i in range(0, len(examples), n)] # 步长为n，分批
			np.random.shuffle(batches) # 打乱批次间的顺序

			log('\nGenerated {} train batches of size {} in {:.3f} sec'.format(len(batches), n, time.time() - start))
			for batch in batches:
				feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch, r)))
				self._session.run(self._enqueue_op, feed_dict=feed_dict)

	def _enqueue_next_test_group(self):
		#Create test batches once and evaluate on them for all test steps
		test_batches, r = self.make_test_batches()
		while not self._coord.should_stop():
			for batch in test_batches:
				feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch, r)))
				self._session.run(self._eval_enqueue_op, feed_dict=feed_dict)

	def _get_next_example(self):
		"""Gets a single example (input, dur_target, token_target, dur_length) from_ disk
		"""
		if self._train_offset >= len(self._train_meta):
			self._train_offset = 0
			np.random.shuffle(self._train_meta)

		meta = self._train_meta[self._train_offset]
		self._train_offset += 1

		score_data = np.load(os.path.join(self._score_dir, meta[4]))
		# score.npy中的存储格式为：['音素所属音符理论时长'，'音符midi', '音素类型', '音素']
		inputs_phoneme = []
		inputs_type = []
		inputs_time = []
		for item in score_data:
			inputs_phoneme.append(self._onehot_encoding(item[3], duration_symbols[0]))
			inputs_type.append(self._onehot_encoding(item[2], duration_symbols[2]))
			inputs_time.append(float(item[0]))			
		
		# duration.npy中的存储格式为：['开始时间', '结束时间', '音素']
		dur_data = np.load(os.path.join(self._dur_dir, meta[3]))
		dur_target = [self._dur_to_target(item) for item in dur_data]
		return (inputs_phoneme, inputs_type, inputs_time, dur_target, len(dur_target))

	def _prepare_batch(self, batches, outputs_per_step):
		assert 0 == len(batches) % self._hparams.tacotron_num_gpus
		size_per_device = int(len(batches) / self._hparams.tacotron_num_gpus)
		np.random.shuffle(batches)

		inputs_phoneme = None
		inputs_time = None
		inputs_type = None
		dur_targets = None
		token_targets = None
		targets_lengths = None
		split_infos = []

		targets_lengths = np.asarray([x[-1] for x in batches], dtype=np.int32) #Used to mask loss
		input_lengths = np.asarray([len(x[0]) for x in batches], dtype=np.int32)

		for i in range(self._hparams.tacotron_num_gpus):
			batch = batches[size_per_device*i:size_per_device*(i+1)]
			input_cur_device_phoneme, input_phoneme_max_len = self._prepare_inputs([x[0] for x in batch])
			input_cur_device_type, input_type_max_len = self._prepare_inputs([x[1] for x in batch])
			input_cur_device_time, input_time_max_len = self._prepare_inputs([x[2] for x in batch])
			inputs_phoneme = np.concatenate((inputs_phoneme, input_cur_device_phoneme), axis=1) if inputs_phoneme is not None else input_cur_device_phoneme
			inputs_type = np.concatenate((inputs_phoneme, input_cur_device_type), axis=1) if inputs_type is not None else input_cur_device_type
			inputs_time = np.concatenate((inputs_time, input_cur_device_time), axis=1) if inputs_time is not None else input_cur_device_time
			dur_target_cur_device, dur_target_max_len = self._prepare_targets([x[3] for x in batch], outputs_per_step)
			dur_targets = np.concatenate(( dur_targets, dur_target_cur_device), axis=1) if dur_targets is not None else dur_target_cur_device

			split_infos.append([input_phoneme_max_len, input_type_max_len, input_time_max_len, dur_target_max_len])

		split_infos = np.asarray(split_infos, dtype=np.int32)
		return (inputs_phoneme, inputs_type, inputs_time, input_lengths, dur_targets, targets_lengths, split_infos)

	# one-hot 编码
	def _onehot_encoding(self, instance, class1):
		temp = [0] * len(class1)
		temp[class1.index(instance)] = 1
		return temp

	def _dur_to_target(self, dur_data):
		return [dur_data[2], float(dur_data[0]), float(dur_data[1])]

	def _prepare_inputs(self, inputs):
		max_len = max([len(x) for x in inputs])
		return np.stack([self._pad_input(x, max_len) for x in inputs]), max_len

	def _prepare_targets(self, targets, alignment):
		max_len = max([len(t) for t in targets])
		data_len = self._round_up(max_len, alignment)
		return np.stack([self._pad_target(t, data_len) for t in targets]), data_len

	def _prepare_token_targets(self, targets, alignment):
		max_len = max([len(t) for t in targets]) + 1
		data_len = self._round_up(max_len, alignment)
		return np.stack([self._pad_token_target(t, data_len) for t in targets]), data_len

	def _pad_input(self, x, length):
		return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=self._pad)

	def _pad_target(self, t, length):
		return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=self._target_pad)

	def _pad_token_target(self, t, length):
		return np.pad(t, (0, length - t.shape[0]), mode='constant', constant_values=self._token_pad)

	def _round_up(self, x, multiple):
		remainder = x % multiple
		return x if remainder == 0 else x + multiple - remainder

	def _round_down(self, x, multiple):
		remainder = x % multiple
		return x if remainder == 0 else x - remainder
