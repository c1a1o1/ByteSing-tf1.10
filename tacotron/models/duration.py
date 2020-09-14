import tensorflow as tf 
from tacotron.utils.symbols import symbols
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

	def initialize(self, inputs, input_lengths, dur_targets=None, stop_token_targets=None, targets_lengths=None,
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
		if dur_targets is None and stop_token_targets is not None:
			raise ValueError('no multi targets were provided but token_targets were given')
		if dur_targets is not None and stop_token_targets is None:
			raise ValueError('Dur targets are provided without corresponding token_targets')
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

			p_inputs = tf.py_func(split_func, [inputs, split_infos[:, 0]], lout_float)
			p_dur_targets = tf.py_func(split_func, [dur_targets, split_infos[:,1]], lout_float) if dur_targets is not None else dur_targets
			p_stop_token_targets = tf.py_func(split_func, [stop_token_targets, split_infos[:,2]], lout_float) if stop_token_targets is not None else stop_token_targets

			tower_inputs = []
			tower_dur_targets = []
			tower_stop_token_targets = []

			batch_size = tf.shape(inputs)[0]
			for i in range (hp.tacotron_num_gpus):
				tower_inputs.append(tf.reshape(p_inputs[i], [batch_size, -1]))
				if p_dur_targets is not None:
					tower_dur_targets.append(tf.reshape(p_dur_targets[i], [batch_size, -1]))
				if p_stop_token_targets is not None:
					tower_stop_token_targets.append(tf.reshape(p_stop_token_targets[i], [batch_size, -1]))

		self.tower_stop_token_prediction = []
		self.tower_dur_outputs = []
		
		# 1. Declare GPU Devices
		gpus = ["/gpu:{}".format(i) for i in range(hp.tacotron_gpu_start_idx, hp.tacotron_gpu_start_idx+hp.tacotron_num_gpus)]
		for i in range(hp.tacotron_num_gpus):
			with tf.device(tf.train.replica_device_setter(ps_tasks=1,ps_device="/cpu:0",worker_device=gpus[i])):
				with tf.variable_scope('inference') as scope:
					# Embeddings ==> [batch_size, sequence_length, embedding_dim]
					self.embedding_table = tf.get_variable(
						'inputs_embedding', [len(symbols), hp.embedding_dim], dtype=tf.float32)
					embedded_inputs = tf.nn.embedding_lookup(self.embedding_table, tower_inputs[i])

					# MultiBiLSTM Cells
					multi_bi_lstm = MultiBiLSTMLayer(is_training, size=hp.multi_bi_lstm_units,
					 zoneout=hp.duration_zoneout_rate, LSTM_layers=hp.duration_layers, scope='duration_MBiLSTM')
					dur_outputs = multi_bi_lstm(tower_inputs[i], tower_input_lengths[i])

					# 加一个dense或2个dense，将维度降为2维，注意最后一层不能加激活函数。

					self.tower_dur_outputs.append(dur_outputs)

			log('initialisation done {}'.format(gpus[i]))



		self.tower_inputs = tower_inputs
		self.tower_input_lengths = tower_input_lengths
		self.tower_dur_targets = tower_dur_targets
		self.tower_targets_lengths = tower_targets_lengths
		self.tower_stop_token_targets = tower_stop_token_targets

		self.all_vars = tf.trainable_variables()

		log('Initialized Duration model. Dimensions (? = dynamic shape): ')
		log('  Train mode:               {}'.format(is_training))
		log('  Eval mode:                {}'.format(is_evaluating))
		log('  Synthesis mode:           {}'.format(not (is_training or is_evaluating)))
		log('  Input:                    {}'.format(inputs.shape))
		for i in range(hp.tacotron_num_gpus+hp.tacotron_gpu_start_idx):
			log('  device:                   {}'.format(i))
			log('  MultiBiLSTM out:          {}'.format(self.tower_dur_outputs[i].shape))

			#1_000_000 is causing syntax problems for some people?! Python please :)
			log('  Duration Parameters       {:.3f} Million.'.format(np.sum([np.prod(v.get_shape().as_list()) for v in self.all_vars]) / 1000000))


	def add_loss(self):
		'''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
		hp = self._hparams

		self.tower_before_loss = []
		self.tower_after_loss= []
		self.tower_stop_token_loss = []
		self.tower_regularization_loss = []
		self.tower_loss = []

		total_before_loss = 0
		total_after_loss= 0
		total_stop_token_loss = 0
		# L2范数不要改
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
						#Compute <stop_token> loss (for learning dynamic generation stop)
						stop_token_loss = MaskedSigmoidCrossEntropy(self.tower_stop_token_targets[i],
							self.tower_stop_token_prediction[i], self.tower_targets_lengths[i], hparams=self._hparams)
					else:
						# Compute loss of predictions before postnet
						before = tf.losses.mean_squared_error(self.tower_mel_targets[i], self.tower_decoder_output[i])
						# Compute loss after postnet
						after = tf.losses.mean_squared_error(self.tower_mel_targets[i], self.tower_mel_outputs[i])
						#Compute <stop_token> loss (for learning dynamic generation stop)
						stop_token_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
							labels=self.tower_stop_token_targets[i],
							logits=self.tower_stop_token_prediction[i]))

						if hp.predict_linear:
							#Compute linear loss
							#From https://github.com/keithito/tacotron/blob/tacotron2-work-in-progress/models/tacotron.py
							#Prioritize loss for frequencies under 2000 Hz.
							l1 = tf.abs(self.tower_linear_targets[i] - self.tower_linear_outputs[i])
							n_priority_freq = int(2000 / (hp.sample_rate * 0.5) * hp.num_freq)
							linear_loss = 0.5 * tf.reduce_mean(l1) + 0.5 * tf.reduce_mean(l1[:,:,0:n_priority_freq])
						else:
							linear_loss = 0.

					# Compute the regularization weight
					if hp.tacotron_scale_regularization:
						reg_weight_scaler = 1. / (2 * hp.max_abs_value) if hp.symmetric_mels else 1. / (hp.max_abs_value)
						reg_weight = hp.tacotron_reg_weight * reg_weight_scaler
					else:
						reg_weight = hp.tacotron_reg_weight

					# Regularize variables
					# Exclude all types of bias, RNN (Bengio et al. On the difficulty of training recurrent neural networks), embeddings and prediction projection layers.
					# Note that we consider attention mechanism v_a weights as a prediction projection layer and we don't regularize it. (This gave better stability)
					regularization = tf.add_n([tf.nn.l2_loss(v) for v in self.all_vars
						if not('bias' in v.name or 'Bias' in v.name or '_projection' in v.name or 'inputs_embedding' in v.name
							or 'RNN' in v.name or 'LSTM' in v.name)]) * reg_weight

					# Compute final loss term
					self.tower_before_loss.append(before)
					self.tower_after_loss.append(after)
					self.tower_stop_token_loss.append(stop_token_loss)
					self.tower_regularization_loss.append(regularization)
					self.tower_linear_loss.append(linear_loss)

					loss = before + after + stop_token_loss + regularization + linear_loss
					self.tower_loss.append(loss)

		for i in range(hp.tacotron_num_gpus):
			total_before_loss += self.tower_before_loss[i] 
			total_after_loss += self.tower_after_loss[i]
			total_stop_token_loss += self.tower_stop_token_loss[i]
			total_regularization_loss += self.tower_regularization_loss[i]
			total_linear_loss += self.tower_linear_loss[i]
			total_loss += self.tower_loss[i]

		self.before_loss = total_before_loss / hp.tacotron_num_gpus
		self.after_loss = total_after_loss / hp.tacotron_num_gpus
		self.stop_token_loss = total_stop_token_loss / hp.tacotron_num_gpus
		self.regularization_loss = total_regularization_loss / hp.tacotron_num_gpus
		self.linear_loss = total_linear_loss / hp.tacotron_num_gpus
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