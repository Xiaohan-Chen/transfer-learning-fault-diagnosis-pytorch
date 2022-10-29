#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch


"""
Created on Saturday Feb 25 2020

@authors: Alan Preciado, Santosh Muthireddy
"""
def CORAL_loss(source, target):
	"""
	From the paper, the vectors that compose Ds and Dt are D-dimensional vectors
	:param source: torch tensor: source data (Ds) with dimensions DxNs
	:param target: torch tensor: target data (Dt) with dimensons DxNt
	"""

	d = source.size(1) # d-dimensional vectors (same for source, target)

	source_covariance = compute_covariance(source)
	target_covariance = compute_covariance(target)

	# take Frobenius norm (https://pytorch.org/docs/stable/torch.html)
	loss = torch.norm(torch.mul((source_covariance-target_covariance),
								(source_covariance-target_covariance)), p="fro")

	# loss = torch.norm(torch.mm((source_covariance-target_covariance),
	# 							(source_covariance-target_covariance)), p="fro")

	loss = loss/(4*d*d)

	return loss


def compute_covariance(data):
	"""
	Compute covariance matrix for given dataset as shown in paper (eqs 2 and 3).
	:param data: torch tensor: input source/target data
	"""

	# data dimensions: nxd (this for Ns or Nt)
	n = data.size(0) # get batch size
	#print("compute covariance bath size n:", n)

  # check gpu or cpu support
	if data.is_cuda:
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	# proper matrix multiplication for right side of equation (2)
	ones_vector = torch.ones(n).resize(1, n).to(device=device) # 1xN dimensional vector (transposed)
	one_onto_D = torch.mm(ones_vector, data)
	mult_right_terms = torch.mm(one_onto_D.t(), one_onto_D)
	mult_right_terms = torch.div(mult_right_terms, n) # element-wise divison

	# matrix multiplication for left side of equation (2)
	mult_left_terms = torch.mm(data.t(), data)

	covariance_matrix= 1/(n-1) * torch.add(mult_left_terms,-1*(mult_right_terms))

	return covariance_matrix
