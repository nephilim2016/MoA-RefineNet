#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 05:06:51 2025

@author: nephilim
"""
import numpy as np
import skimage.transform
import tensorflow as tf


def normalization_Sigmoid(data):
    min_ = np.min(data)
    max_ = np.max(data)
    N_data = (data - min_) / (max_ - min_)
    return N_data, min_, max_


def normalization_Tanh(data):
    N_data, min_, max_ = normalization_Sigmoid(data)
    N_data = 2 * (N_data - 0.5)
    return N_data, min_, max_


def re_normalization_Sigmoid(N_data, min_, max_):
    data = N_data * (max_ - min_) + min_
    return data


def re_normalization_Tanh(N_data, min_, max_):
    data = N_data / 2 + 0.5
    data = re_normalization_Sigmoid(data, min_, max_)
    return data


def nextstep(step, length):
    return step * np.ceil(length / step).astype(int)


def get_horizontal_blocks(data, block_size=256, step=32):
    blocks_list = []
    blocks = np.zeros((data.shape[0], block_size))
    for idx_col in range(0, data.shape[1] - block_size + 1, step):
        blocks = data[:, idx_col : idx_col + block_size]
        blocks_list.append(blocks)
    return blocks_list


def stack_horizontal_blocks(data, blocks, block_size=256, step=32):
    data = np.zeros_like(data)
    weight = np.zeros_like(data)
    idx = 0
    for block in blocks:
        data[:, idx : idx + block_size] += block
        weight[:, idx : idx + block_size] += 1
        idx += step
    return data, weight


def get_vertical_blocks(data, block_size=256, step=32):
    blocks_list = []
    blocks = np.zeros((block_size, data.shape[1]))
    for idx_row in range(0, data.shape[0] - block_size + 1, step):
        blocks = data[idx_row : idx_row + block_size, :]
        blocks_list.append(blocks)
    return blocks_list


def stack_vertical_blocks(data, blocks, block_size=256, step=32):
    data = np.zeros_like(data)
    weight = np.zeros_like(data)
    idx = 0
    for block in blocks:
        data[idx : idx + block_size, :] += block
        weight[idx : idx + block_size, :] += 1
        idx += step
    return data, weight


def block_list(data, step):
    if data.shape[1] < 256:
        data = skimage.transform.resize(data, (256, 256))
    else:
        data = skimage.transform.resize(data, (256, nextstep(step, data.shape[1])))
    data_blocks_list = get_horizontal_blocks(data, block_size=256, step=step)
    return data_blocks_list


def inference_patch(
    model, inference_data, resize_shape, block_size=256, step=32, eps=1e-6
):
    inference_data = skimage.transform.resize(inference_data, resize_shape)
    inference_data, _, _ = normalization_Tanh(inference_data)
    vertical_blocks = get_vertical_blocks(
        inference_data, block_size=block_size, step=step
    )
    blocks_list = []
    for vertical_block in vertical_blocks:
        horizontal_blocks = get_horizontal_blocks(
            vertical_block, block_size=block_size, step=step
        )
        blocks_list.append(horizontal_blocks)
    prediction_list = []
    for block_list in blocks_list:
        inference_block = np.array(block_list)[..., np.newaxis]
        prediction = model(tf.convert_to_tensor(inference_block), training=False)
        prediction = prediction.numpy()
        prediction -= np.mean(prediction)
        prediction = prediction[:, :, :, 0]
        prediction_list.append(list(prediction))
    stack_vertical_blocks_list = []
    for horizontal_block in prediction_list:
        stack_vertical_block, stack_vertical_block_weight = stack_horizontal_blocks(
            np.zeros((block_size, inference_data.shape[1])),
            horizontal_block,
            block_size=block_size,
            step=step,
        )
        stack_vertical_blocks_list.append(
            stack_vertical_block / (stack_vertical_block_weight + eps)
        )
    stack_blocks, stack_blocks_weights = stack_vertical_blocks(
        np.zeros((inference_data.shape[0], inference_data.shape[1])),
        stack_vertical_blocks_list,
        block_size=block_size,
        step=step,
    )
    prediction = stack_blocks / (stack_blocks_weights + eps)
    return inference_data, prediction
