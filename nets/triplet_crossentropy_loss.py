import tensorflow as tf
from tensorflow_addons.losses import metric_learning
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike
from typeguard import typechecked
from typing import Optional

def _get_triplet_mask(labels):
	"""Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
	A triplet (i, j, k) is valid if:
		- i, j, k are distinct
		- labels[i] == labels[j] and labels[i] != labels[k]
	Args:
		labels: tf.int32 `Tensor` with shape [batch_size]
	"""
	# Check that i, j and k are distinct
	indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
	indices_not_equal = tf.logical_not(indices_equal)
	i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
	i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
	j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

	distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


	# Check if labels[i] == labels[j] and labels[i] != labels[k]
	label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
	i_equal_j = tf.expand_dims(label_equal, 2)
	i_equal_k = tf.expand_dims(label_equal, 1)

	valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

	# Combine the two masks
	mask = tf.logical_and(distinct_indices, valid_labels)

	return mask


@tf.function
def triplet_batchall_crossentropy_loss(
	y_true: TensorLike, y_pred: TensorLike, margin: FloatTensorLike = 1.0
) -> tf.Tensor:
	"""Computes the triplet loss with semi-hard negative mining.

	Args:
	  y_true: 1-D integer `Tensor` with shape [batch_size] of
		multiclass integer labels.
	  y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should
		be l2 normalized.
	  margin: Float, margin term in the loss definition.

	Returns:
	  triplet_loss: float scalar with dtype of y_pred.
	"""
	labels, embeddings = y_true, y_pred

	convert_to_float32 = (
			embeddings.dtype == tf.dtypes.float16 or embeddings.dtype == tf.dtypes.bfloat16
	)
	precise_embeddings = (
		tf.cast(embeddings, tf.dtypes.float32) if convert_to_float32 else embeddings
	)

	# Reshape label tensor to [batch_size, 1].
	lshape = tf.shape(labels)
	labels = tf.reshape(labels, [lshape[0], 1])

	# Build pairwise squared distance matrix.
	pdist_matrix = metric_learning.pairwise_distance(precise_embeddings, squared=True)

	# shape (batch_size, batch_size, 1)
	anchor_positive_dist = tf.expand_dims(pdist_matrix, 2)

	# shape (batch_size, 1, batch_size)
	anchor_negative_dist = tf.expand_dims(pdist_matrix, 1)

	# Compute a 3D tensor of size (batch_size, batch_size, batch_size)
	# triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
	# Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
	# and the 2nd (batch_size, 1, batch_size)
	distances = anchor_positive_dist - anchor_negative_dist + margin

	# Put to zero the invalid triplets
	# (where label(a) != label(p) or label(n) == label(a) or a == p)
	mask = _get_triplet_mask(labels)
	mask = tf.cast(mask, tf.dtypes.float32)
	distances = tf.multiply(mask, distances)

	# Count number of positive triplets (where triplet_loss > 0)
	valid_triplets = tf.cast(tf.greater(distances, 1e-16), tf.dtypes.float32)
	num_positive_triplets = tf.reduce_sum(valid_triplets)

	# Get final mean triplet loss over the valid triplets
	triplet_loss = tf.math.truediv(
		tf.math.reduce_sum(
			tf.math.multiply(tf.nn.softplus(distances), valid_triplets)
		),
		num_positive_triplets + 1e-16,
	)

	if convert_to_float32:
		return tf.cast(triplet_loss, embeddings.dtype)
	else:
		return triplet_loss

class TripletBatchAllCrossentropyLoss(tf.keras.losses.Loss):
	"""Computes the triplet loss with semi-hard negative mining.
	The loss encourages the positive distances (between a pair of embeddings
	with the same labels) to be smaller than the minimum negative distance
	among which are at least greater than the positive distance plus the
	margin constant (called semi-hard negative) in the mini-batch.
	If no such negative exists, uses the largest negative distance instead.
	See: https://arxiv.org/abs/1503.03832.
	We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
	[batch_size] of multi-class integer labels. And embeddings `y_pred` must be
	2-D float `Tensor` of l2 normalized embedding vectors.
	Args:
	  margin: Float, margin term in the loss definition. Default value is 1.0.
	  name: Optional name for the op.
	"""

	@typechecked
	def __init__(
		self, margin: FloatTensorLike = 1.0, name: Optional[str] = None, **kwargs
	):
		super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)
		self.margin = margin

	def call(self, y_true, y_pred):
		return triplet_batchall_crossentropy_loss(y_true, y_pred, self.margin)

	def get_config(self):
		config = {
			"margin": self.margin,
		}
		base_config = super().get_config()
		return {**base_config, **config}


def _masked_maximum(data, mask, dim=1):
	"""Computes the axis wise maximum over chosen elements.

	Args:
	  data: 2-D float `Tensor` of size [n, m].
	  mask: 2-D Boolean `Tensor` of size [n, m].
	  dim: The dimension over which to compute the maximum.

	Returns:
	  masked_maximums: N-D `Tensor`.
		The maximized dimension is of size 1 after the operation.
	"""
	axis_minimums = tf.math.reduce_min(data, dim, keepdims=True)
	masked_maximums = (
		tf.math.reduce_max(
			tf.math.multiply(data - axis_minimums, mask), dim, keepdims=True
		)
		+ axis_minimums
	)
	return masked_maximums


def _masked_minimum(data, mask, dim=1):
	"""Computes the axis wise minimum over chosen elements.

	Args:
	  data: 2-D float `Tensor` of size [n, m].
	  mask: 2-D Boolean `Tensor` of size [n, m].
	  dim: The dimension over which to compute the minimum.

	Returns:
	  masked_minimums: N-D `Tensor`.
		The minimized dimension is of size 1 after the operation.
	"""
	axis_maximums = tf.math.reduce_max(data, dim, keepdims=True)
	masked_minimums = (
		tf.math.reduce_min(
			tf.math.multiply(data - axis_maximums, mask), dim, keepdims=True
		)
		+ axis_maximums
	)
	return masked_minimums


@tf.function
def triplet_semihard_crossentropy_loss(
	y_true: TensorLike, y_pred: TensorLike, margin: FloatTensorLike = 1.0
) -> tf.Tensor:
	"""Computes the triplet loss with semi-hard negative mining.

	Args:
	  y_true: 1-D integer `Tensor` with shape [batch_size] of
		multiclass integer labels.
	  y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should
		be l2 normalized.
	  margin: Float, margin term in the loss definition.

	Returns:
	  triplet_loss: float scalar with dtype of y_pred.
	"""
	labels, embeddings = y_true, y_pred

	convert_to_float32 = (
		embeddings.dtype == tf.dtypes.float16 or embeddings.dtype == tf.dtypes.bfloat16
	)
	precise_embeddings = (
		tf.cast(embeddings, tf.dtypes.float32) if convert_to_float32 else embeddings
	)

	# Reshape label tensor to [batch_size, 1].
	lshape = tf.shape(labels)
	labels = tf.reshape(labels, [lshape[0], 1])

	# Build pairwise squared distance matrix.
	pdist_matrix = metric_learning.pairwise_distance(precise_embeddings, squared=True)
	# Build pairwise binary adjacency matrix.
	adjacency = tf.math.equal(labels, tf.transpose(labels))
	# Invert so we can select negatives only.
	adjacency_not = tf.math.logical_not(adjacency)

	batch_size = tf.size(labels)

	# Compute the mask.
	pdist_matrix_tile = tf.tile(pdist_matrix, [batch_size, 1])
	mask = tf.math.logical_and(
		tf.tile(adjacency_not, [batch_size, 1]),
		tf.math.greater(
			pdist_matrix_tile, tf.reshape(tf.transpose(pdist_matrix), [-1, 1])
		),
	)
	mask_final = tf.reshape(
		tf.math.greater(
			tf.math.reduce_sum(
				tf.cast(mask, dtype=tf.dtypes.float32), 1, keepdims=True
			),
			0.0,
		),
		[batch_size, batch_size],
	)
	mask_final = tf.transpose(mask_final)

	adjacency_not = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
	mask = tf.cast(mask, dtype=tf.dtypes.float32)

	# negatives_outside: smallest D_an where D_an > D_ap.
	negatives_outside = tf.reshape(
		_masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size]
	)
	negatives_outside = tf.transpose(negatives_outside)

	# negatives_inside: largest D_an.
	negatives_inside = tf.tile(
		_masked_maximum(pdist_matrix, adjacency_not), [1, batch_size]
	)
	semi_hard_negatives = tf.where(mask_final, negatives_outside, negatives_inside)

	loss_mat = tf.math.add(margin, pdist_matrix - semi_hard_negatives)

	mask_positives = tf.cast(adjacency, dtype=tf.dtypes.float32) - tf.linalg.diag(
		tf.ones([batch_size])
	)

	loss_mat = tf.math.maximum(tf.math.multiply(loss_mat, mask_positives), 0.0)
	final_mask = tf.cast(tf.greater(loss_mat, 1e-16), tf.dtypes.float32)
	final_positives = tf.reduce_sum(final_mask)

	triplet_loss = tf.math.truediv(
		tf.math.reduce_sum(
			tf.math.multiply(tf.nn.softplus(loss_mat), final_mask)
		),
		final_positives,
	)

	if convert_to_float32:
		return tf.cast(triplet_loss, embeddings.dtype)
	else:
		return triplet_loss


@tf.function
def triplet_hard_crossentropy_loss(
	y_true: TensorLike,
	y_pred: TensorLike,
	margin: FloatTensorLike = 1.0,
) -> tf.Tensor:
	"""Computes the triplet loss with hard negative and hard positive mining.

	Args:
	  y_true: 1-D integer `Tensor` with shape [batch_size] of
		multiclass integer labels.
	  y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should
		be l2 normalized.
	  margin: Float, margin term in the loss definition.

	Returns:
	  triplet_loss: float scalar with dtype of y_pred.
	"""
	labels, embeddings = y_true, y_pred

	convert_to_float32 = (
		embeddings.dtype == tf.dtypes.float16 or embeddings.dtype == tf.dtypes.bfloat16
	)
	precise_embeddings = (
		tf.cast(embeddings, tf.dtypes.float32) if convert_to_float32 else embeddings
	)

	# Reshape label tensor to [batch_size, 1].
	lshape = tf.shape(labels)
	labels = tf.reshape(labels, [lshape[0], 1])

	# Build pairwise squared distance matrix.
	pdist_matrix = metric_learning.pairwise_distance(precise_embeddings, squared=True)
	# Build pairwise binary adjacency matrix.
	adjacency = tf.math.equal(labels, tf.transpose(labels))
	# Invert so we can select negatives only.
	adjacency_not = tf.math.logical_not(adjacency)

	adjacency_not = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
	# hard negatives: smallest D_an.
	hard_negatives = _masked_minimum(pdist_matrix, adjacency_not)

	batch_size = tf.size(labels)

	adjacency = tf.cast(adjacency, dtype=tf.dtypes.float32)

	mask_positives = tf.cast(adjacency, dtype=tf.dtypes.float32) - tf.linalg.diag(
		tf.ones([batch_size])
	)

	# hard positives: largest D_ap.
	hard_positives = _masked_maximum(pdist_matrix, mask_positives)

	loss_mat = tf.maximum(hard_positives - hard_negatives + margin, 0.0)
	final_mask = tf.cast(tf.greater(loss_mat, 1e-16), tf.dtypes.float32)
	final_positives = tf.reduce_sum(final_mask)

	# Get final mean triplet loss
	triplet_loss = tf.math.truediv(
		tf.math.reduce_sum(
			tf.math.multiply(tf.nn.softplus(loss_mat), final_mask)
		),
		final_positives,
	)

	if convert_to_float32:
		return tf.cast(triplet_loss, embeddings.dtype)
	else:
		return triplet_loss


class TripletSemiHardCrossentropyLoss(tf.keras.losses.Loss):
	"""Computes the triplet loss with semi-hard negative mining.

	The loss encourages the positive distances (between a pair of embeddings
	with the same labels) to be smaller than the minimum negative distance
	among which are at least greater than the positive distance plus the
	margin constant (called semi-hard negative) in the mini-batch.
	If no such negative exists, uses the largest negative distance instead.
	See: https://arxiv.org/abs/1503.03832.

	We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
	[batch_size] of multi-class integer labels. And embeddings `y_pred` must be
	2-D float `Tensor` of l2 normalized embedding vectors.

	Args:
	  margin: Float, margin term in the loss definition. Default value is 1.0.
	  name: Optional name for the op.
	"""

	@typechecked
	def __init__(
		self, margin: FloatTensorLike = 1.0, name: Optional[str] = None, **kwargs
	):
		super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)
		self.margin = margin

	def call(self, y_true, y_pred):
		return triplet_semihard_crossentropy_loss(y_true, y_pred, self.margin)

	def get_config(self):
		config = {
			"margin": self.margin,
		}
		base_config = super().get_config()
		return {**base_config, **config}


class TripletHardCrossentropyLoss(tf.keras.losses.Loss):
	"""Computes the triplet loss with hard negative and hard positive mining.

	The loss encourages the maximum positive distance (between a pair of embeddings
	with the same labels) to be smaller than the minimum negative distance plus the
	margin constant in the mini-batch.
	The loss selects the hardest positive and the hardest negative samples
	within the batch when forming the triplets for computing the loss.
	See: https://arxiv.org/pdf/1703.07737.

	We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
	[batch_size] of multi-class integer labels. And embeddings `y_pred` must be
	2-D float `Tensor` of l2 normalized embedding vectors.

	Args:
	  margin: Float, margin term in the loss definition. Default value is 1.0.
	  soft: Boolean, if set, use the soft margin version. Default value is False.
	  name: Optional name for the op.
	"""

	@typechecked
	def __init__(
		self,
		margin: FloatTensorLike = 1.0,
		name: Optional[str] = None,
		**kwargs
	):
		super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)
		self.margin = margin

	def call(self, y_true, y_pred):
		return triplet_hard_crossentropy_loss(y_true, y_pred, self.margin)

	def get_config(self):
		config = {
			"margin": self.margin,
		}
		base_config = super().get_config()
		return {**base_config, **config}


if __name__ == '__main__':
	import numpy as np
	logits = tf.convert_to_tensor([[1.1, 1.2, 1.4], [1.09, 1.21,1.41], [0.25, 0.45, 0.75], [0.23, 0.43, 0.7], [1.5, 2.5, 3.5], [1.55, 2.75, 3.8]], dtype=tf.dtypes.float32)
	labels = tf.convert_to_tensor(np.array([1, 1, 2, 2, 3, 3]), dtype=tf.dtypes.float32)
	loss = triplet_batchall_crossentropy_loss(labels, logits)
	print(loss)

