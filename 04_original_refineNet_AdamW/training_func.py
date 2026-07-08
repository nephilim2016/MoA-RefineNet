#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 05:15:57 2025

@author: nephilim
"""
import tensorflow as tf
from loss_func import reconstruciton_loss, entropy_loss, kl_to_uniform_batch_mean, entropy_norm, entropy_to_target


def collect_vars_top(backbone):
    vars_top = []
    top_names = ["output_stage1", "output_cbam1", "output_stage2", "output_cbam2"]
    for name in top_names:
        try:
            layer = backbone.get_layer(name)
            vars_top += layer.trainable_variables
        except ValueError:
            pass
    return vars_top


@tf.function
def compute_apply_gradients_backbone(model, x, y, optimizer):
    with tf.GradientTape() as tape:
        predict = model(x, training=True)
        loss, loss_charbonnier, loss_ps = reconstruciton_loss(predict, y, max_val=2.0)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return {"loss": loss, "loss_charb": loss_charbonnier, "loss_ps": loss_ps}


@tf.function
def compute_apply_gradients_moa(
    model, x, y, optimizer, lambda_sparse, lambda_diversity
):
    with tf.GradientTape() as tape:
        predict = model(x, training=True)
        loss_rec, _, _ = reconstruciton_loss(predict, y, max_val=2.0)

        # === routing regularization ===
        w = getattr(model.moa, "last_w", None)
        if w is not None:
            w_clip = tf.clip_by_value(w, 1e-8, 1.0)
            loss_sparse = entropy_loss(w_clip)
            loss_Ht = entropy_to_target(w, H_target=tf.constant(0.2, w.dtype))
            loss_diversity = kl_to_uniform_batch_mean(w)
            w_mean = tf.reduce_mean(w, axis=0)
            top1 = tf.reduce_mean(tf.reduce_max(w, axis=-1))
            H_bar = tf.reduce_mean(entropy_norm(w))
        else:
            loss_sparse = tf.constant(0.0, tf.float32)
            loss_diversity = tf.constant(0.0, tf.float32)
            loss_Ht = tf.constant(0.0, tf.float32)
            w_mean = tf.zeros([1], tf.float32)
            top1 = tf.constant(0.0, tf.float32)
            H_bar = tf.constant(0.0, tf.float32)
        loss = (
            loss_rec + lambda_sparse * loss_sparse + lambda_diversity * loss_diversity + + 0.5 * loss_Ht
        )

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return {
        "loss": loss,
        "loss_rec": loss_rec,
        "loss_sparse": loss_sparse,
        "loss_diversity": loss_diversity,
        "w_mean": w_mean,
        "top1": top1,
        "w": w,
        "H_bar": H_bar,
    }


@tf.function
def compute_apply_gradients_moa_part_backbone(
    model, x, y, optimizer_moa, optimizer_top, lambda_sparse, lambda_diversity
):
    with tf.GradientTape() as tape:
        predict = model(x, training=True)
        loss_rec, _, _ = reconstruciton_loss(predict, y, max_val=2.0)

        # === routing regularization ===
        w = getattr(model.moa, "last_w", None)
        if w is not None:
            w_clip = tf.clip_by_value(w, 1e-8, 1.0)
            loss_sparse = entropy_loss(w_clip)
            loss_Ht = entropy_to_target(w, H_target=tf.constant(0.2, w.dtype))
            loss_diversity = kl_to_uniform_batch_mean(w)
            w_mean = tf.reduce_mean(w, axis=0)
            top1 = tf.reduce_mean(tf.reduce_max(w, axis=-1))
            H_bar = tf.reduce_mean(entropy_norm(w))
        else:
            loss_sparse = tf.constant(0.0, tf.float32)
            loss_diversity = tf.constant(0.0, tf.float32)
            loss_Ht = tf.constant(0.0, tf.float32)
            w_mean = tf.zeros([1], tf.float32)
            top1 = tf.constant(0.0, tf.float32)
            H_bar = tf.constant(0.0, tf.float32)
        loss = (
            loss_rec + lambda_sparse * loss_sparse + lambda_diversity * loss_diversity + + 0.5 * loss_Ht
        )

    vars_moa = model.moa.trainable_variables
    vars_top = collect_vars_top(model.backbone)

    train_vars = vars_moa + vars_top
    grads = tape.gradient(loss, train_vars)

    gradients_moa = grads[: len(vars_moa)]
    gradients_top = grads[len(vars_moa) :]

    optimizer_moa.apply_gradients(zip(gradients_moa, vars_moa))
    optimizer_top.apply_gradients(zip(gradients_top, vars_top))

    return {
        "loss": loss,
        "loss_rec": loss_rec,
        "loss_sparse": loss_sparse,
        "loss_diversity": loss_diversity,
        "w_mean": w_mean,
        "top1": top1,
        "w": w,
        "H_bar": H_bar,
    }


@tf.function
def compute_backbone(model, x, y):
    predict = model(x, training=False)
    loss, loss_charbonnier, loss_ps = reconstruciton_loss(predict, y, max_val=2.0)
    return {"loss": loss, "loss_charb": loss_charbonnier, "loss_ps": loss_ps}


@tf.function
def compute_moa(model, x, y, lambda_sparse, lambda_diversity):
    predict = model(x, training=False)
    loss_rec, _, _ = reconstruciton_loss(predict, y, max_val=2.0)
    # === routing regularization ===
    w = getattr(model.moa, "last_w", None)
    if w is not None:
        w_clip = tf.clip_by_value(w, 1e-8, 1.0)
        loss_sparse = entropy_loss(w_clip)
        loss_diversity = kl_to_uniform_batch_mean(w)
        w_mean = tf.reduce_mean(w, axis=0)
        top1 = tf.reduce_mean(tf.reduce_max(w, axis=-1))
        H_bar = tf.reduce_mean(entropy_norm(w))
    else:
        loss_sparse = tf.constant(0.0, tf.float32)
        loss_diversity = tf.constant(0.0, tf.float32)
        w_mean = tf.zeros([1], tf.float32)
        top1 = tf.constant(0.0, tf.float32)
        H_bar = tf.constant(0.0, tf.float32)
    loss = loss_rec + lambda_sparse * loss_sparse + lambda_diversity * loss_diversity
    return {
        "loss": loss,
        "loss_rec": loss_rec,
        "loss_sparse": loss_sparse,
        "loss_diversity": loss_diversity,
        "w_mean": w_mean,
        "top1": top1,
        "w": w,
        "H_bar": H_bar,
    }


@tf.function
def compute_moa_part_backbone(model, x, y, lambda_sparse, lambda_diversity):
    predict = model(x, training=False)
    loss_rec, _, _ = reconstruciton_loss(predict, y, max_val=2.0)
    # === routing regularization ===
    w = getattr(model.moa, "last_w", None)
    if w is not None:
        w_clip = tf.clip_by_value(w, 1e-8, 1.0)
        loss_sparse = entropy_loss(w_clip)
        loss_diversity = kl_to_uniform_batch_mean(w)
        w_mean = tf.reduce_mean(w, axis=0)
        top1 = tf.reduce_mean(tf.reduce_max(w, axis=-1))
        H_bar = tf.reduce_mean(entropy_norm(w))
    else:
        loss_sparse = tf.constant(0.0, tf.float32)
        loss_diversity = tf.constant(0.0, tf.float32)
        w_mean = tf.zeros([1], tf.float32)
        top1 = tf.constant(0.0, tf.float32)
        H_bar = tf.constant(0.0, tf.float32)
    loss = loss_rec + lambda_sparse * loss_sparse + lambda_diversity * loss_diversity
    return {
        "loss": loss,
        "loss_rec": loss_rec,
        "loss_sparse": loss_sparse,
        "loss_diversity": loss_diversity,
        "w_mean": w_mean,
        "top1": top1,
        "w": w,
        "H_bar": H_bar,
    }
