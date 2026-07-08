#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 00:25:50 2025

@author: nephilim
"""

import os
import numpy as np
from matplotlib import pyplot,cm
import tensorflow as tf
import skimage.transform
import Normalization

def nextstep(step,length):
    return step*np.ceil(length/step).astype(int)

def get_horizontal_blocks(data,block_size=256,step=32):
    blocks_list=[]
    blocks=np.zeros((data.shape[0],block_size))
    for idx_col in range(0,data.shape[1]-block_size+1,step):
        blocks=data[:,idx_col:idx_col+block_size]
        blocks_list.append(blocks)
    return blocks_list

def stack_horizontal_blocks(data,blocks,block_size=256,step=32):
    data=np.zeros_like(data)
    weight=np.zeros_like(data)
    idx=0
    for block in blocks:
        data[:,idx:idx+block_size]+=block
        weight[:,idx:idx+block_size]+=1
        idx+=step
    return data,weight

def get_vertical_blocks(data,block_size=256,step=32):
    blocks_list=[]
    blocks=np.zeros((block_size,data.shape[1]))
    for idx_row in range(0,data.shape[0]-block_size+1,step):
        blocks=data[idx_row:idx_row+block_size,:]
        blocks_list.append(blocks)
    return blocks_list

def stack_vertical_blocks(data,blocks,block_size=256,step=32):
    data=np.zeros_like(data)
    weight=np.zeros_like(data)
    idx=0
    for block in blocks:
        data[idx:idx+block_size,:]+=block
        weight[idx:idx+block_size,:]+=1
        idx+=step
    return data,weight

def block_list(data,step):
    if data.shape[1]<256:
        data=skimage.transform.resize(data,(256,256))
    else:
        data=skimage.transform.resize(data,(256,nextstep(step,data.shape[1])))
    data_blocks_list=get_horizontal_blocks(data,block_size=256,step=step)
    return data_blocks_list

def inference_patch(model,inference_data,resize_shape,block_size=256,step=32,eps=1e-6):
    inference_data=skimage.transform.resize(inference_data,resize_shape)
    inference_data,_,_=Normalization.NormalizationTanh(inference_data)
    vertical_blocks=get_vertical_blocks(inference_data,block_size=block_size,step=step)
    blocks_list=[]
    for vertical_block in vertical_blocks:
        horizontal_blocks=get_horizontal_blocks(vertical_block,block_size=block_size,step=step)
        blocks_list.append(horizontal_blocks)
    prediction_list=[]
    for block_list in blocks_list:
        inference_block=np.array(block_list)[...,np.newaxis]
        prediction=model(tf.convert_to_tensor(inference_block),training=False)
        prediction=prediction.numpy()
        prediction-=np.mean(prediction)
        prediction=prediction[:,:,:,0]
        prediction_list.append(list(prediction))
    stack_vertical_blocks_list=[]
    for horizontal_block in prediction_list:
        stack_vertical_block,stack_vertical_block_weight=stack_horizontal_blocks(np.zeros((block_size,inference_data.shape[1])),horizontal_block,block_size=block_size,step=step)
        stack_vertical_blocks_list.append(stack_vertical_block/(stack_vertical_block_weight+eps))
    stack_blocks,stack_blocks_weights=stack_vertical_blocks(np.zeros((inference_data.shape[0],inference_data.shape[1])),stack_vertical_blocks_list,block_size=block_size,step=step)
    prediction=stack_blocks/(stack_blocks_weights+eps)
    return inference_data,prediction

if __name__=='__main__':
    test_input_clutter=np.load('00 clutter_testing.npy').astype('float32')
    test_input_crts=np.load('01 crts_testing.npy').astype('float32')
    test_input_rebar1=np.load('02 rebar_testing.npy').astype('float32')
    test_input_rebar2=np.load('03 rebar_testing.npy').astype('float32')
    
    model=tf.keras.models.load_model('./Weights/Stage2_model_50')
    
    clutter_inference,clutter_prediction=inference_patch(model,test_input_clutter,resize_shape=(256,256),block_size=256,step=2)
    crts_inference,crts_prediction=inference_patch(model,test_input_crts,resize_shape=(256,256),block_size=256,step=2)
    rebar1_inference,rebar1_prediction=inference_patch(model,test_input_rebar1,resize_shape=(256,256),block_size=256,step=2)
    rebar2_inference,rebar2_prediction=inference_patch(model,test_input_rebar2,resize_shape=(256,256),block_size=256,step=2)
    
    pyplot.figure()
    pyplot.subplot(221)
    pyplot.imshow(clutter_inference,cmap=cm.gray,extent=(0,2,1,0),vmin=-1,vmax=1)
    pyplot.subplot(222)
    pyplot.imshow(crts_inference,cmap=cm.gray,extent=(0,2,1,0),vmin=-1,vmax=1)
    pyplot.subplot(223)
    pyplot.imshow(rebar1_inference,cmap=cm.gray,extent=(0,2,1,0),vmin=-1,vmax=1)
    pyplot.subplot(224)
    pyplot.imshow(rebar2_inference,cmap=cm.gray,extent=(0,2,1,0),vmin=-1,vmax=1)
    
    pyplot.figure()
    pyplot.subplot(221)
    pyplot.imshow(clutter_prediction,cmap=cm.gray,extent=(0,2,1,0),vmin=-1,vmax=1)
    pyplot.subplot(222)
    pyplot.imshow(crts_prediction,cmap=cm.gray,extent=(0,2,1,0),vmin=-1,vmax=1)
    pyplot.subplot(223)
    pyplot.imshow(rebar1_prediction,cmap=cm.gray,extent=(0,2,1,0),vmin=-1,vmax=1)
    pyplot.subplot(224)
    pyplot.imshow(rebar2_prediction,cmap=cm.gray,extent=(0,2,1,0),vmin=-1,vmax=1)
    