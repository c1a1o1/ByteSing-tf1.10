import tensorflow as tf 
from tacotron.utils.symbols import duration_symbols
from infolog import log
from tacotron.models.helpers import TacoTrainingHelper, TacoTestHelper
from tacotron.models.modules import *
from tensorflow.contrib.seq2seq import dynamic_decode
from tacotron.models.Architecture_wrappers import TacotronEncoderCell, TacotronDecoderCell
from tacotron.models.custom_decoder import CustomDecoder
from tacotron.models.attention import LocationSensitiveAttention

import numpy as np

def split_func(x, split_pos):
	rst = []
	start = 0
	# x will be a numpy array with the contents of the placeholder below
	for i in range(split_pos.shape[0]):
		rst.append(x[:,start:start+split_pos[i]])
		start += split_pos[i]
	return rst

class Duration():
	"""Duration prediction Model.
	"""
	def __init__(self, hparams):
		self._hparams = hparams

	def initialize(self, inputs_phoneme, inputs_type, inputs_time, input_lengths, dur_targets=None, targets_lengths=None,
			global_step=None, is_training=False, is_evaluating=False, split_infos=None):
		"""
		Initializes the model for inference
		sets "mel_outputs" and "alignments" fields.
		Args:
			- inputs: int32 Tensor with shape [N, T_in] where N is batch size, T_in is number of
			  steps in the input time series, and values are character IDs
			- input_lengths: int32 Tensor with shape [N] where N is batch size and values are the lengths
			of each sequence in inputs.
			- mel_targets: float32 Tensor with shape [N, T_out, M] where N is batch size, T_out is number
			of steps in the output time series, M is num_mels, and values are entries in the mel
			spectrogram. Only needed for training.
		"""
		if dur_targets is None:
			raise ValueError('no multi targets were provided but token_targets were given')
		if is_training and self._hparams.mask_decoder and targets_lengths is None:
			raise RuntimeError('Model set to mask paddings but no targets lengths provided for the mask!')
		if is_training and is_evaluating:
			raise RuntimeError('Model can not be in training and evaluation modes at the same time!')

		split_device = '/cpu:0' if self._hparams.tacotron_num_gpus > 1 or self._hparams.split_on_cpu else '/gpu:{}'.format(self._hparams.tacotron_gpu_start_idx)
		with tf.device(split_device):
			hp = self._hparams
			lout_int = [tf.int32]*hp.tacotron_num_gpus
			lout_float = [tf.float32]*hp.tacotron_num_gpus

			tower_input_lengths = tf.split(input_lengths, num_or_size_splits=hp.tacotron_num_gpus, axis=0)
			tower_targets_lengths = tf.split(targets_lengths, num_or_size_splits=hp.tacotron_num_gpus, axis=0) if targets_lengths is not None else targets_lengths

			p_inputs_phoneme = tf.py_func(split_func, [inputs_phoneme, split_infos[:, 0]], lout_float)
			p_inputs_type = tf.py_func(split_func, [inputs_type, split_infos[:, 1]], lout_float)
			p_inputs_time = tf.py_func(split_func, [inputs_time, split_infos[:, 2]], lout_float)
			p_dur_targets = tf.py_func(split_func, [dur_targets, split_infos[:, 3]], lout_float) if dur_targets is not None else dur_targets

			tower_inputs_phoneme = []
			tower_inputs_type = []
			tower_inputs_time = []
			tower_dur_targets = []

			batch_size = tf.shape(tower_inputs_phoneme)[0]
			for i in range (hp.tacotron_num_gpus):
				tower_inputs_phoneme.append(tf.reshape(p_inputs_phoneme[i], [batch_size, -1]))
				tower_inputs_type.append(tf.reshape(p_inputs_type[i], [batch_size, -1]))
				tower_inputs_time.append(tf.reshape(p_inputs_time[i], [batch_size, -1]))
				if p_dur_targets is not None:
					tower_dur_targets.append(tf.reshape(p_dur_targets[i], [batch_size, -1]))


		self.tower_dur_outputs = []
		tower_embedded_inputs = []
		
		# 1. Declare GPU Devices
		gpus = ["/gpu:{}".format(i) for i in range(hp.tacotron_gpu_start_idx, hp.tacotron_gpu_start_idx+hp.tacotron_num_gpus)]
		for i in range(hp.tacotron_num_gpus):
			with tf.device(tf.train.replica_device_setter(ps_tasks=1,ps_device="/cpu:0",worker_device=gpus[i])):
				with tf.variable_scope('inference') as scope:
					# Embeddings ==> [batch_size, sequence_length, embedding_dim]
					self.phoneme_embedding_table = tf.get_variable(
						'inputs_phoneme_embedding', [len(duration_symbols[0]), hp.duration_phoneme_embedding_dim], dtype=tf.float32)					
					self.type_embedding_table = tf.get_variable(
						'inputs_type_embedding', [len(duration_symbols[2]), hp.duration_type_embedding_dim], dtype=tf.float32)
					embedded_inputs_phoneme = tf.nn.embedding_lookup(self.phoneme_embedding_table, tower_inputs_phoneme[i])
					embedded_inputs_type = tf.nn.embedding_lookup(self.type_embedding_table, tower_inputs_type[i])
					
					tower_embedded_inputs = tf.concat([embedded_inputs_phoneme, embedded_inputs_type], axis = -1)
					tower_embedded_inputs = tf.concat([tower_embedded_inputs, tower_inputs_time], axis = -1)

					# MultiBiLSTM Cells
					multi_bi_lstm = MultiBiLSTMLayer(is_training, size=hp.multi_bi_lstm_units,
					 zoneout=hp.duration_zoneout_rate, LSTM_layers=hp.duration_layers, scope='duration_MBiLSTM')
					mb_lstm_outputs = multi_bi_lstm(tower_embedded_inputs, tower_input_lengths[i])

					# 加一个dense或2个dense，将维度降为2维，注意最后一层不能加激活函数。
					dense1 = tf.layers.dense(inputs=mb_lstm_outputs, units = 3, activation = None)
					self.tower_dur_outputs.append(dense1)

			log('initialisation done {}'.format(gpus[i]))



		self.tower_inputs_phoneme = tower_inputs_phoneme
		self.tower_inputs_type = tower_inputs_type
		self.tower_inputs_time = tower_inputs_time
		self.tower_input_lengths = tower_input_lengths
		self.tower_dur_targets = tower_dur_targets
		self.tower_targets_lengths = tower_targets_lengths

		self.all_vars = tf.trainable_variables()

		log('Initialized Duration model. Dimensions (? = dynamic shape): ')
		log('  Train mode:               {}'.format(is_training))
		log('  Eval mode:                {}'.format(is_evaluating))
		log('  Synthesis mode:           {}'.format(not (is_training or is_evaluating)))
		log('  Input:                    {}'.format(inputs_phoneme.shape))
		for i in range(hp.tacotron_num_gpus+hp.tacotron_gpu_start_idx):
			log('  device:                   {}'.format(i))
			log('  MultiBiLSTM out:          {}'.format(self.tower_dur_outputs[i].shape))

			#1_000_000 is causing syntax problems for some people?! Python please :)
			log('  Duration Parameters       {:.3f} Million.'.format(np.sum([np.prod(v.get_shape().as_list()) for v in self.all_vars]) / 1000000))


	def add_loss(self):
		'''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
		hp = self._hparams

		self.tower_before_loss = []
		self.tower_loss = []
		self.tower_regularization_loss = []

		# L2范数不要改
		total_before_loss = 0
		total_regularization_loss = 0
		total_loss = 0

		gpus = ["/gpu:{}".format(i) for i in range(hp.tacotron_gpu_start_idx, hp.tacotron_gpu_start_idx+hp.tacotron_num_gpus)]

		for i in range(hp.tacotron_num_gpus):
			with tf.device(tf.train.replica_device_setter(ps_tasks=1,ps_device="/cpu:0",worker_device=gpus[i])):
				with tf.variable_scope('loss') as scope:
					if hp.mask_duration:
						# Compute loss of predictions
						before = MaskedMSE(self.tower_dur_targets[i], self.tower_dur_outputs[i], self.tower_targets_lengths[i],
							hparams=self._hparams)
					else:
						# Compute loss of predictions before postnet
						before = tf.losses.mean_squared_error(self.tower_dur_targets[i], self.tower_dur_outputs[i])

					# Compute the regularization weight
					if hp.tacotron_scale_regularization:
						reg_weight_scaler = 1. / (2 * hp.max_abs_value) if hp.symmetric_mels else 1. / (hp.max_abs_value)
						reg_weight = hp.tacotron_reg_weight * reg_weight_scaler
					else:
						reg_weight = hp.tacotron_reg_weight

					# Regularize variables
					# Exclude all types of bias, RNN (Bengio et al. On the difficulty of training recurrent neural networks), embeddings and prediction projection layers.
					# Note that we consider attention mechanism v_a weights as a prediction projection layer and we don't regularize it. (This gave better stability)
					regularization = tf.add_n([tf.nn.l2_loss(v) for v in self.all_vars]) * reg_weight

					# Compute final loss term
					self.tower_before_loss.append(before)
					self.tower_regularization_loss.append(regularization)

					loss = before + regularization
					self.tower_loss.append(loss)

		for i in range(hp.tacotron_num_gpus):
			total_before_loss += self.tower_before_loss[i] 
			total_regularization_loss += self.tower_regularization_loss[i]
			total_loss += self.tower_loss[i]

		self.before_loss = total_before_loss / hp.tacotron_num_gpus
		self.regularization_loss = total_regularization_loss / hp.tacotron_num_gpus
		self.loss = total_loss / hp.tacotron_num_gpus

	def add_optimizer(self, global_step):
		'''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.
		Args:
			global_step: int32 scalar Tensor representing current global step in training
		'''
		hp = self._hparams
		tower_gradients = []

		# 1. Declare GPU Devices
		gpus = ["/gpu:{}".format(i) for i in range(hp.tacotron_gpu_start_idx, hp.tacotron_gpu_start_idx + hp.tacotron_num_gpus)]

		grad_device = '/cpu:0' if hp.tacotron_num_gpus > 1 else gpus[0]

		with tf.device(grad_device):
			with tf.variable_scope('optimizer') as scope:
				if hp.tacotron_decay_learning_rate:
					self.decay_steps = hp.tacotron_decay_steps
					self.decay_rate = hp.tacotron_decay_rate
					self.learning_rate = self._learning_rate_decay(hp.tacotron_initial_learning_rate, global_step)
				else:
					self.learning_rate = tf.convert_to_tensor(hp.tacotron_initial_learning_rate)

				optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.tacotron_adam_beta1,
					hp.tacotron_adam_beta2, hp.tacotron_adam_epsilon)

		# 2. Compute Gradient
		for i in range(hp.tacotron_num_gpus):
			#  Device placement
			with tf.device(tf.train.replica_device_setter(ps_tasks=1,ps_device="/cpu:0",worker_device=gpus[i])) :
				#agg_loss += self.tower_loss[i]
				with tf.variable_scope('optimizer') as scope:
					gradients = optimizer.compute_gradients(self.tower_loss[i])
					tower_gradients.append(gradients)

		# 3. Average Gradient
		with tf.device(grad_device) :
			avg_grads = []
			vars = []
			for grad_and_vars in zip(*tower_gradients):
				# grads_vars = [(grad1, var), (grad2, var), ...]
				grads = []
				for g,_ in grad_and_vars:
					expanded_g = tf.expand_dims(g, 0)
					# Append on a 'tower' dimension which we will average over below.
					grads.append(expanded_g)
					# Average over the 'tower' dimension.
				grad = tf.concat(axis=0, values=grads)
				grad = tf.reduce_mean(grad, 0)

				v = grad_and_vars[0][1]
				avg_grads.append(grad)
				vars.append(v)

			self.gradients = avg_grads
			#Just for causion
			#https://github.com/Rayhane-mamah/Tacotron-2/issues/11
			if hp.tacotron_clip_gradients:
				clipped_gradients, _ = tf.clip_by_global_norm(avg_grads, 1.) # __mark 0.5 refer
			else:
				clipped_gradients = avg_grads

			# Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
			# https://github.com/tensorflow/tensorflow/issues/1122
			with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
				self.optimize = optimizer.apply_gradients(zip(clipped_gradients, vars),
					global_step=global_step)

	def _learning_rate_decay(self, init_lr, global_step):
		#################################################################
		# Narrow Exponential Decay:

		# Phase 1: lr = 1e-3
		# We only start learning rate decay after 50k steps

		# Phase 2: lr in ]1e-5, 1e-3[
		# decay reach minimal value at step 310k

		# Phase 3: lr = 1e-5
		# clip by minimal learning rate value (step > 310k)
		#################################################################
		hp = self._hparams

		#Compute natural exponential decay
		lr = tf.train.exponential_decay(init_lr, 
			global_step - hp.tacotron_start_decay, #lr = 1e-3 at step 50k
			self.decay_steps, 
			self.decay_rate, #lr = 1e-5 around step 310k
			name='lr_exponential_decay')


		#clip learning rate by max and min values (initial and final values)
		return tf.minimum(tf.maximum(lr, hp.tacotron_final_learning_rate), init_lr)