#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 19:30:51 2022

@author: nephilim
"""
import numpy as np

def NormalizationSigmoid(data):
    min_=np.min(data)
    max_=np.max(data)
    N_data=(data-min_)/(max_-min_)
    return N_data,min_,max_

def NormalizationTanh(data):
    N_data,min_,max_=NormalizationSigmoid(data)
    N_data=2*(N_data-0.5)
    return N_data,min_,max_

def ReNormalizationSigmoid(N_data,min_,max_):
    data=N_data*(max_-min_)+min_
    return data

def ReNormalizationTanh(N_data,min_,max_):
    data=N_data/2+0.5
    data=ReNormalizationSigmoid(data,min_,max_)
    return data