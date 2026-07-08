#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 05:13:19 2025

@author: nephilim
"""
import tensorflow as tf


def reconstruciton_loss(y_pred, y_true, max_val=2.0):
    loss_ssim = ssim_loss(y_pred, y_true, max_val=max_val)
    loss_psnr = psnr_loss(y_pred, y_true, max_val=max_val)
    loss_ps = loss_ssim / (loss_psnr + 1e-8)
    loss_charbonnier = charbonnier_loss(y_pred, y_true, eps=1e-8)
    loss = loss_charbonnier + loss_ps * 5
    return loss, loss_charbonnier, loss_ps


def charbonnier_loss(y_pred, y_true, eps=1e-3):
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + eps**2))


def psnr_loss(y_pred, y_true, max_val=2.0, eps=1e-8):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    mse = tf.maximum(mse, eps)
    return 10.0 * tf.math.log((max_val**2) / mse) / tf.math.log(10.0)


def ssim_loss(y_pred, y_true, max_val=2.0):
    loss_ssim = tf.image.ssim(y_true, y_pred, max_val=max_val)
    loss_ssim = 1 - tf.reduce_mean(loss_ssim)
    return loss_ssim


def entropy_loss(w, eps=1e-8):
    w = tf.clip_by_value(w, eps, 1.0)
    ent = -tf.reduce_sum(w * tf.math.log(w), axis=-1)  # [B]
    return tf.reduce_mean(ent)


def diversity_loss(w):
    w_mean = tf.reduce_mean(w, axis=0)  # [E]
    E = tf.cast(tf.shape(w_mean)[0], tf.float32)
    u = tf.ones_like(w_mean) / E
    return tf.reduce_mean(tf.square(w_mean - u))


def entropy_norm(w, eps=1e-8):
    w = tf.clip_by_value(w, eps, 1.0)
    H = -tf.reduce_sum(w * tf.math.log(w), axis=-1)  # [B]
    E = tf.cast(tf.shape(w)[-1], w.dtype)
    return H / tf.math.log(E)  # [B] ∈ [0,1]


def entropy_to_target(w, H_target):
    Hn = entropy_norm(w)
    return tf.reduce_mean(tf.square(Hn - H_target))


def kl_to_uniform_batch_mean(w, eps=1e-8):
    w_mean = tf.reduce_mean(w, axis=0)  # [E]
    w_mean = tf.clip_by_value(w_mean, eps, 1.0)
    E = tf.cast(tf.shape(w_mean)[0], w_mean.dtype)
    return tf.reduce_sum(w_mean * (tf.math.log(w_mean) + tf.math.log(E)))
