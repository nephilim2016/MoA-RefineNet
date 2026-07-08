#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 12:48:40 2025

@author: nephilim
"""

import tensorflow as tf
from tensorflow.keras import layers, initializers


# %%
# micro residual adapter: 1×1 dimension-reduction → k×k → 1×1 dimension-expansion (very small)
class ResidualBottleneck(layers.Layer):
    def __init__(self, channels, k=3, r_ratio=0.25, name=None):
        super().__init__(name=name)
        mid = max(1, int(channels * r_ratio))
        self.conv1 = layers.Conv2D(
            mid, 1, padding="same", use_bias=True, kernel_initializer="he_normal"
        )
        self.act1 = layers.LeakyReLU()
        # self.act1 = layers.ReLU()
        self.conv2 = layers.Conv2D(
            mid, k, padding="same", use_bias=True, kernel_initializer="he_normal"
        )
        self.act2 = layers.LeakyReLU()
        # self.act2 = layers.ReLU()
        # initialize the last layer with zeros so that, when inserted, it behaves close to identity — stabilizes training
        self.conv3 = layers.Conv2D(
            channels,
            1,
            padding="same",
            use_bias=True,
            kernel_initializer=initializers.RandomNormal(stddev=1e-3),
        )

    def call(self, x, training=None):
        y = self.conv1(x, training=training)
        y = self.act1(y)
        y = self.conv2(y, training=training)
        y = self.act2(y)
        y = self.conv3(y, training=training)
        return y


# %%
# Router: GAP → small MLP → softmax weights
class MoARouter(tf.keras.layers.Layer):
    def __init__(self, num_experts=3, hidden=64, tau_init=2.0, name=None):
        super().__init__(name=name)
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(hidden, activation="relu")
        self.fc2 = tf.keras.layers.Dense(
            num_experts, bias_initializer=initializers.RandomNormal(stddev=1e-3)
        )
        # Adjustable temperature (non-trainable variable, assigned externally from the training loop)
        self.tau = tf.Variable(tau_init, trainable=False, dtype=tf.float32)
        # Runtime configuration (controlled by external setter)
        self._use_topk = (
            0  # 0 means no top-k; >0 means top-k with k equal to this value
        )
        self._use_gumbel = False
        # cache, used for regularization and logging
        self.last_w_soft = None
        self.last_w_hard = None

    # Convenience setter
    def set_tau(self, tau: float):
        self.tau.assign(float(tau))

    def set_topk(self, k: int | None):
        self._use_topk = int(k) if k is not None else 0

    def set_gumbel(self, flag: bool):
        self._use_gumbel = bool(flag)

    def call(self, x, training=None):
        # GAP -> MLP -> logits
        h = self.fc1(self.gap(x), training=training)
        z = self.fc2(h, training=training)  # [B,E] logits

        # Optionally inject Gumbel noise during training to encourage exploration
        if training and self._use_gumbel:
            u = tf.random.uniform(tf.shape(z), 1e-6, 1 - 1e-6)
            g = -tf.math.log(-tf.math.log(u))
            z = z + g

        # Temperature-scaled softmax (soft weights: used for statistics / regularization)
        w_soft = tf.nn.softmax(z / self.tau, axis=-1)  # [B,E]
        ######################################################
        eps = 0.05  # 前期 0.05，后期可降到 0.01
        E = tf.cast(tf.shape(w_soft)[-1], w_soft.dtype)
        w_soft = (1.0 - eps) * w_soft + eps / E
        ######################################################
        self.last_w_soft = w_soft

        # Mixture weights: optional hard top-k during training (for forward mixture)
        w = w_soft
        if training and self._use_topk > 0:
            k = tf.minimum(self._use_topk, tf.shape(w_soft)[-1])
            vals, idx = tf.math.top_k(w_soft, k=k)
            mask = tf.reduce_sum(
                tf.one_hot(idx, depth=tf.shape(w_soft)[-1]), axis=1
            )  # [B,E]
            w_hard = tf.stop_gradient(mask) + w_soft - tf.stop_gradient(w_soft)
            w = w_hard / (tf.reduce_sum(w_hard, axis=-1, keepdims=True) + 1e-8)
            self.last_w_head = w
        else:
            self.last_w_hard = None
        return w


# %%
# 3) MoA residual block: E adapters + soft routing + residual fusion
class MoAResidualBlock(layers.Layer):
    """
    Usage:
      y, w = MoAResidualBlock(C, num_experts=3, r_ratio=0.25, k=3)(x, training=True)
    """

    def __init__(
        self,
        channels,
        num_experts=3,
        r_ratio=0.25,
        k=3,
        router_hidden=64,
        tau_init=1.5,
        name="moa_block",
    ):
        super().__init__(name=name)
        self.adapters = [
            ResidualBottleneck(
                channels, k=k, r_ratio=r_ratio, name=f"{name}_adapter{i}"
            )
            for i in range(num_experts)
        ]
        self.router = MoARouter(
            num_experts=num_experts,
            hidden=router_hidden,
            tau_init=tau_init,
            name=f"{name}_router",
        )
        self.num_experts = num_experts

        # Conveniently allows extracting routing weights during training for regularization / visualization
        self.last_w = None

    def _rms_norm(self, y, eps=1e-6):
        # Normalize by RMS over (B,H,W,C) to keep comparable scales — prevents one expert “winning” just by larger magnitude
        rms = tf.sqrt(tf.reduce_mean(tf.square(y), axis=[1, 2, 3], keepdims=True) + eps)
        return y / rms

    def call(self, x, training=None):
        # During training, use the router’s built-in top-k / τ / Gumbel settings
        w = self.router(x, training=training)  # [B,E]
        # For statistics/regularization always use soft weights (to avoid hard routing causing H=0)
        self.last_w = getattr(self.router, "last_w_soft", w)

        outs = []
        for i, a in enumerate(self.adapters):
            y_i = a(x, training=training)  # [B,H,W,C]
            y_i = self._rms_norm(y_i)  # prevent magnitude bias
            w_i = tf.reshape(w[:, i], (-1, 1, 1, 1))  # [B,1,1,1]
            outs.append(w_i * y_i)

        y = x + (tf.add_n(outs) if len(outs) > 1 else outs[0])
        return y, w
