#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch


"""
Created on Sunday 22 Mar 2020

@authors: Alan Preciado, Santosh Muthireddy
"""
def MMDLinear(source_activation, target_activation):
	"""
	From the paper, the loss used is the maximum mean discrepancy (MMD)
	:param source: torch tensor: source data (Ds) with dimensions DxNs
	:param target: torch tensor: target data (Dt) with dimensons DxNt
	"""

	diff_domains = source_activation - target_activation
	loss = torch.mean(torch.mm(diff_domains, torch.transpose(diff_domains, 0, 1)))

	return loss
