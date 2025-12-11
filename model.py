import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import conv

from torch_geometric.nn import (
	Aggregation,
	MaxAggregation,
	MeanAggregation,
	MultiAggregation,
	SAGEConv,
	SoftmaxAggregation,
	StdAggregation,
	SumAggregation,
	VarAggregation,
	SetTransformerAggregation,
	PowerMeanAggregation,
	DeepGCNLayer,
	GINEConv,
	DenseSAGEConv,
	LSTMAggregation,
	GraphSAGE,
	LayerNorm,
	GATv2Conv,
	GAT,
	GPSConv,
	TransformerConv,
	global_add_pool
)

from torch_geometric.data import Data

from torch.nn import (
	ReLU,
	Sequential,
	BatchNorm1d,
	Embedding,
	ModuleList
)

import torch_geometric.transforms as T

from typing import Dict

import copy
from typing import Any, List, Optional, Union

import numpy as np
import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric.typing import Adj


class InvertibleFunction(torch.autograd.Function):
	r"""An invertible autograd function. This allows for automatic
	backpropagation in a reversible fashion so that the memory of intermediate
	results can be freed during the forward pass and be constructed on-the-fly
	during the bachward pass.

	Args:
		ctx (torch.autograd.function.InvertibleFunctionBackward):
			A context object that can be used to stash information for backward
			computation.
		fn (torch.nn.Module): The forward function.
		fn_inverse (torch.nn.Module): The inverse function to recompute the
			freed input.
		num_bwd_passes (int): Number of backward passes to retain a link
			with the output. After the last backward pass the output is
			discarded and memory is freed.
		num_inputs (int): The number of inputs to the forward function.
		*args (tuple): Inputs and weights.
	"""
	@staticmethod
	def forward(ctx, fn: torch.nn.Module, fn_inverse: torch.nn.Module,
				num_bwd_passes: int, num_inputs: int, *args):
		ctx.fn = fn
		ctx.fn_inverse = fn_inverse
		ctx.weights = args[num_inputs:]
		ctx.num_bwd_passes = num_bwd_passes
		ctx.num_inputs = num_inputs
		inputs = args[:num_inputs]
		ctx.input_requires_grad = []

		with torch.no_grad():  # Make a detached copy which shares the storage:
			x = []
			for element in inputs:
				if isinstance(element, torch.Tensor):
					x.append(element.detach())
					ctx.input_requires_grad.append(element.requires_grad)
				else:
					x.append(element)
					ctx.input_requires_grad.append(None)
			outputs = ctx.fn(*x)

		if not isinstance(outputs, tuple):
			outputs = (outputs, )

		# Detaches outputs in-place, allows discarding the intermedate result:
		detached_outputs = tuple(element.detach_() for element in outputs)

		# Clear memory of node features:
		if torch_geometric.typing.WITH_PT20:
			inputs[0].cpu().untyped_storage().resize_(0)
		else:  # pragma: no cover
			inputs[0].storage().resize_(0)

		# Store these tensor nodes for backward passes:
		ctx.inputs = [inputs] * num_bwd_passes
		ctx.outputs = [detached_outputs] * num_bwd_passes

		return detached_outputs

	@staticmethod
	def backward(ctx, *grad_outputs):
		if len(ctx.outputs) == 0:
			raise RuntimeError(
				f"Trying to perform a backward pass on the "
				f"'InvertibleFunction' for more than '{ctx.num_bwd_passes}' "
				f"times. Try raising 'num_bwd_passes'.")

		inputs = ctx.inputs.pop()
		outputs = ctx.outputs.pop()

		# Recompute input by swapping out the first argument:
		with torch.no_grad():
			inputs_inverted = ctx.fn_inverse(*(outputs + inputs[1:]))
			if len(ctx.outputs) == 0:  # Clear memory from outputs:
				for element in outputs:
					if torch_geometric.typing.WITH_PT20:
						element.cpu().untyped_storage().resize_(0)
					else:  # pragma: no cover
						element.storage().resize_(0)

			if not isinstance(inputs_inverted, tuple):
				inputs_inverted = (inputs_inverted, )

			for elem_orig, elem_inv in zip(inputs, inputs_inverted):
				if torch_geometric.typing.WITH_PT20:
					elem_orig.cpu().untyped_storage().resize_(
						int(np.prod(elem_orig.size())) *
						elem_orig.element_size())
				else:  # pragma: no cover
					elem_orig.storage().resize_(int(np.prod(elem_orig.size())))
				elem_orig.set_(elem_inv)

		# Compute gradients with grad enabled:
		with torch.set_grad_enabled(True):
			detached_inputs = []
			for element in inputs:
				if isinstance(element, torch.Tensor):
					detached_inputs.append(element.detach())
				else:
					detached_inputs.append(element)
			detached_inputs = tuple(detached_inputs)
			for x, req_grad in zip(detached_inputs, ctx.input_requires_grad):
				if isinstance(x, torch.Tensor):
					x.requires_grad = req_grad
			tmp_output = ctx.fn(*detached_inputs)

		if not isinstance(tmp_output, tuple):
			tmp_output = (tmp_output, )

		filtered_detached_inputs = tuple(
			filter(
				lambda x: x.requires_grad
				if isinstance(x, torch.Tensor) else False,
				detached_inputs,
			))
		gradients = torch.autograd.grad(
			outputs=tmp_output,
			inputs=filtered_detached_inputs + ctx.weights,
			grad_outputs=grad_outputs,
		)

		input_gradients = []
		i = 0
		for rg in ctx.input_requires_grad:
			if rg:
				input_gradients.append(gradients[i])
				i += 1
			else:
				input_gradients.append(None)

		gradients = tuple(input_gradients) + gradients[-len(ctx.weights):]

		return (None, None, None, None) + gradients


class InvertibleModule(torch.nn.Module):
	r"""An abstract class for implementing invertible modules.

	Args:
		disable (bool, optional): If set to :obj:`True`, will disable the usage
			of :class:`InvertibleFunction` and will execute the module without
			memory savings. (default: :obj:`False`)
		num_bwd_passes (int, optional): Number of backward passes to retain a
			link with the output. After the last backward pass the output is
			discarded and memory is freed. (default: :obj:`1`)
	"""
	def __init__(self, disable: bool = False, num_bwd_passes: int = 1):
		super().__init__()
		self.disable = disable
		self.num_bwd_passes = num_bwd_passes

	def forward(self, *args):
		""""""  # noqa: D419
		return self._fn_apply(args, self._forward, self._inverse)

	def inverse(self, *args):
		return self._fn_apply(args, self._inverse, self._forward)

	def _forward(self):
		raise NotImplementedError

	def _inverse(self):
		raise NotImplementedError

	def _fn_apply(self, args, fn, fn_inverse):
		if not self.disable:
			out = InvertibleFunction.apply(
				fn,
				fn_inverse,
				self.num_bwd_passes,
				len(args),
				*args,
				*tuple(p for p in self.parameters() if p.requires_grad),
			)
		else:
			out = fn(*args)

		# If the layer only has one input, we unpack the tuple:
		if isinstance(out, tuple) and len(out) == 1:
			return out[0]

		return out


class GroupAddRev(InvertibleModule):
	r"""The Grouped Reversible GNN module from the `"Graph Neural Networks with
	1000 Layers" <https://arxiv.org/abs/2106.07476>`_ paper.
	This module enables training of arbitrary deep GNNs with a memory
	complexity independent of the number of layers.

	It does so by partitioning input node features :math:`\mathbf{X}` into
	:math:`C` groups across the feature dimension. Then, a grouped reversible
	GNN block :math:`f_{\theta(i)}` operates on a group of inputs and produces
	a group of outputs:

	.. math::

		\mathbf{X}^{\prime}_0 &= \sum_{i=2}^C \mathbf{X}_i

		\mathbf{X}^{\prime}_i &= f_{\theta(i)} ( \mathbf{X}^{\prime}_{i - 1},
		\mathbf{A}) + \mathbf{X}_i

	for all :math:`i \in \{ 1, \ldots, C \}`.

	.. note::

		For an example of using :class:`GroupAddRev`, see `examples/rev_gnn.py
		<https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
		rev_gnn.py>`_.

	Args:
		conv (torch.nn.Module or torch.nn.ModuleList]): A seed GNN. The input
			and output feature dimensions need to match.
		split_dim (int, optional): The dimension across which to split groups.
			(default: :obj:`-1`)
		num_groups (int, optional): The number of groups :math:`C`.
			(default: :obj:`None`)
		disable (bool, optional): If set to :obj:`True`, will disable the usage
			of :class:`InvertibleFunction` and will execute the module without
			memory savings. (default: :obj:`False`)
		num_bwd_passes (int, optional): Number of backward passes to retain a
			link with the output. After the last backward pass the output is
			discarded and memory is freed. (default: :obj:`1`)
	"""
	def __init__(
		self,
		conv: Union[torch.nn.Module, torch.nn.ModuleList],
		split_dim: int = -1,
		num_groups: Optional[int] = None,
		disable: bool = False,
		num_bwd_passes: int = 1,
	):
		super().__init__(disable, num_bwd_passes)
		self.split_dim = split_dim

		if isinstance(conv, torch.nn.ModuleList):
			self.convs = conv
		else:
			assert num_groups is not None, "Please specific 'num_groups'"
			self.convs = torch.nn.ModuleList([conv])
			for _ in range(num_groups - 1):
				conv = copy.deepcopy(self.convs[0])
				if hasattr(conv, 'reset_parameters'):
					conv.reset_parameters()
				self.convs.append(conv)

		if len(self.convs) < 2:
			raise ValueError(f"The number of groups should not be smaller "
							 f"than '2' (got '{self.num_groups}'))")

	@property
	def num_groups(self) -> int:
		return len(self.convs)

	def reset_parameters(self):
		r"""Resets all learnable parameters of the module."""
		for conv in self.convs:
			conv.reset_parameters()

	def _forward(self, x: Tensor, edge_index: Adj, *args):
		channels = x.size(self.split_dim)
		xs = self._chunk(x, channels)
		args = list(zip(*[self._chunk(arg, channels) for arg in args]))
		args = [[]] * self.num_groups if len(args) == 0 else args

		ys = []
		y_in = sum(xs[1:])
		for i in range(self.num_groups):
			y_in = xs[i] + self.convs[i](y_in, edge_index, *args[i])
			ys.append(y_in)
		return torch.cat(ys, dim=self.split_dim)

	def _inverse(self, y: Tensor, edge_index: Adj, *args):
		channels = y.size(self.split_dim)
		ys = self._chunk(y, channels)
		args = list(zip(*[self._chunk(arg, channels) for arg in args]))
		args = [[]] * self.num_groups if len(args) == 0 else args

		xs = []
		for i in range(self.num_groups - 1, -1, -1):
			if i != 0:
				y_in = ys[i - 1]
			else:
				y_in = sum(xs)
			x = ys[i] - self.convs[i](y_in, edge_index, *args[i])
			xs.append(x)

		return torch.cat(xs[::-1], dim=self.split_dim)

	def _chunk(self, x: Any, channels: int) -> List[Any]:
		if not isinstance(x, Tensor):
			return [x] * self.num_groups

		try:
			if x.size(self.split_dim) != channels:
				return [x] * self.num_groups
		except IndexError:
			return [x] * self.num_groups

		return torch.chunk(x, self.num_groups, dim=self.split_dim)

	def __repr__(self) -> str:
		return (f'{self.__class__.__name__}({self.convs[0]}, '
				f'num_groups={self.num_groups})')


class GNNBlock(torch.nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.norm = LayerNorm(in_channels, affine=True)
		self.conv = SAGEConv(in_channels, out_channels)

	def reset_parameters(self):
		self.norm.reset_parameters()
		self.conv.reset_parameters()

	def forward(self, x, edge_index, dropout_mask=None):
		x = self.norm(x).relu()
		if self.training and dropout_mask is not None:
			x = x * dropout_mask
		return self.conv(x, edge_index)


class RevGNN(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
				 dropout, num_groups=2):
		super().__init__()

		self.dropout = dropout

		self.lin1 = Linear(in_channels, hidden_channels)
		self.lin2 = Linear(hidden_channels, out_channels)
		self.norm = LayerNorm(hidden_channels, affine=True)

		assert hidden_channels % num_groups == 0
		self.convs = torch.nn.ModuleList()
		for _ in range(num_layers):
			conv = GNNBlock(
				hidden_channels // num_groups,
				hidden_channels // num_groups,
			)
			self.convs.append(GroupAddRev(conv, num_groups=num_groups))

	def reset_parameters(self):
		self.lin1.reset_parameters()
		self.lin2.reset_parameters()
		self.norm.reset_parameters()
		for conv in self.convs:
			conv.reset_parameters()

	def forward(self, x, edge_index):
		x = self.lin1(x)

		# Generate a dropout mask which will be shared across GNN blocks:
		mask = None
		if self.training and self.dropout > 0:
			mask = torch.zeros_like(x).bernoulli_(1 - self.dropout)
			mask = mask.requires_grad_(False)
			mask = mask / (1 - self.dropout)

		for conv in self.convs:
			x = conv(x, edge_index, mask)
		x = self.norm(x).relu()
		x = F.dropout(x, p=self.dropout, training=self.training)
		return self.lin2(x)

class GraphTransformer(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
		super().__init__()
		self.pe_dim = 8

		combined_in_channels = in_channels + self.pe_dim

		self.conv1 = TransformerConv(combined_in_channels, hidden_channels, heads=heads)
		self.conv2 = TransformerConv(hidden_channels * heads, hidden_channels, heads=heads)
		self.conv3 = TransformerConv(hidden_channels * heads, out_channels, heads=1)

	def forward(self, x, edge_index, pe):
		x = torch.cat([x, pe], dim=1)

		x = F.relu(self.conv1(x, edge_index))
		x = F.dropout(x, p=0.2, training=self.training)
		x = F.relu(self.conv2(x, edge_index))
		x = F.dropout(x, p=0.2, training=self.training)
		x = self.conv3(x, edge_index)

		return x

class CustomGNN(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(CustomGNN, self).__init__()
		#self.prelayer1 = nn.Linear(input_dim, hidden_dim)
		#self.prelayer2 = nn.Linear(hidden_dim, hidden_dim)
		#self.prelayer3 = nn.Linear(hidden_dim, hidden_dim)
		#self.layers = GAT(input_dim, hidden_dim, 2, output_dim, 0.2, act="tanh")
		#self.layers = GraphSAGE(hidden_dim, hidden_dim, 3, output_dim, 0.2, act="tanh", aggr=LSTMAggregation(hidden_dim, hidden_dim))
		#self.layers = RevGNN(input_dim, hidden_dim, output_dim, 10, 0.1, 2)
		#self.layer1 = GPSConv(input_dim, hidden_dim)
		#self.layerH1 = GPSConv(hidden_dim, hidden_dim)
		#self.layer2 = GPSConv(hidden_dim, output_dim)
		self.layers = GraphTransformer(
			in_channels=input_dim,
			hidden_channels=hidden_dim,
			out_channels=output_dim,
			heads=8
		)
		#self.layer1 = SAGEConv(input_dim, hidden_dim, aggr=LSTMAggregation(input_dim, input_dim))
		#self.layerH1 = SAGEConv(hidden_dim, hidden_dim, aggr=LSTMAggregation(hidden_dim, hidden_dim))
		#self.layer2 = SAGEConv(hidden_dim, output_dim, aggr=LSTMAggregation(hidden_dim, hidden_dim))
		#self.postlayer1 = nn.Linear(hidden_dim, hidden_dim)
		#self.postlayer2 = nn.Linear(hidden_dim, hidden_dim)
		#self.postlayer3 = nn.Linear(hidden_dim, output_dim)

		self.edge_weight = nn.Sequential(nn.Linear(output_dim * 2, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
		self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

	def forward(self, feature_data, edge_info):
		row, col = edge_info
		perm = col.argsort()
		edge_info = edge_info[:, perm]

		transform = T.AddLaplacianEigenvectorPE(k=8, attr_name='pe')
		feature_data = transform(Data(feature_data, edge_info)).to(self.device)


		#x = self.prelayer1(feature_data).relu()
		#x = F.dropout(x, p=0.2, training=self.training)
		#x = self.prelayer2(x).relu()
		#x = F.dropout(x, p=0.2, training=self.training)
		#x = self.prelayer3(x).relu()
		#x = F.dropout(x, p=0.2, training=self.training)
		#x = self.layers(feature_data, edge_info)
		#x = self.layer1(feature_data, edge_info).relu()
		#x = F.dropout(x, p=0.2, training=self.training)
		#x = self.layerH1(x, edge_info).relu()
		#x = F.dropout(x, p=0.2, training=self.training)
		#x = self.layer2(x, edge_info).tanh()
		self.layers.training = self.training
		x = self.layers(feature_data.x, feature_data.edge_index, feature_data.pe)
		#x = F.dropout(x, p=0.2, training=self.training)
		#x = self.postlayer1(x).relu()
		#x = F.dropout(x, p=0.2, training=self.training)
		#x = self.postlayer2(x).relu()
		#x = F.dropout(x, p=0.2, training=self.training)
		#x = self.postlayer3(x).tanh()

		#embedding_a = x

		return x, x
		"""N, d = x.shape

		x_i = x.unsqueeze(1).expand(N, N, d)
		x_j = x.unsqueeze(0).expand(N, N, d)

		pairs = torch.cat((x_i, x_j), dim=-1)

		pairs_flat = pairs.reshape(N * N, -1)

		scores = self.edge_weight(pairs_flat).tanh()

		embedding = scores.view(N, N)

		embedding = embedding * (1 - torch.eye(N, device=x.device)) + torch.eye(N, device=x.device)

		embedding = (embedding + embedding.transpose(-1, -2)) / 2

		#embedding = torch.linalg.eigh(embedding).eigenvalues

		#embedding = torch.sort(embedding, dim=-1).values

		return embedding_a, embedding"""
	
def fully_connected_edges(n_nodes, device):
	"""Return edge_index for a fully connected directed graph WITHOUT self-loops."""
	src = torch.arange(n_nodes, device=device).repeat_interleave(n_nodes)
	dst = torch.arange(n_nodes, device=device).repeat(n_nodes)
	
	mask = src != dst
	src, dst = src[mask], dst[mask]
	
	edge_index = torch.stack([src, dst], dim=0)
	return edge_index