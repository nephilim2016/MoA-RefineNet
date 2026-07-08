#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 12:48:40 2025

@author: nephilim
"""

import tensorflow as tf
from tensorflow.keras import layers, initializers

# 1) 微型残差适配器：1x1降维 -> kxk -> 1x1升维（很小）
class ResidualBottleneck(layers.Layer):
    def __init__(self, channels, k=3, r_ratio=0.25, name=None):
        super().__init__(name=name)
        mid = max(1, int(channels * r_ratio))
        self.conv1 = layers.Conv2D(mid, 1, padding="same", use_bias=True,
                                   kernel_initializer="he_normal")
        self.act1  = layers.ReLU()
        self.conv2 = layers.Conv2D(mid, k, padding="same", use_bias=True,
                                   kernel_initializer="he_normal")
        self.act2  = layers.ReLU()
        # 以 0 初始化最后一层，确保刚插入时“近似恒等”，训练更稳
        self.conv3 = layers.Conv2D(channels, 1, padding="same", use_bias=True,
                                   kernel_initializer=initializers.RandomNormal(stddev=1e-3))

    def call(self, x, training=None):
        y = self.conv1(x, training=training); y = self.act1(y)
        y = self.conv2(y, training=training); y = self.act2(y)
        y = self.conv3(y, training=training)
        return y

class MoARouter(tf.keras.layers.Layer):
    def __init__(self, num_experts=3, hidden=64, tau_init=2.0, name=None):
        super().__init__(name=name)
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(hidden, activation='relu')
        self.fc2 = tf.keras.layers.Dense(
            num_experts,
            bias_initializer=initializers.RandomNormal(stddev=1e-3)
        )
        # 可调温度（不可训练变量，由训练循环外部 assign）
        self.tau = tf.Variable(tau_init, trainable=False, dtype=tf.float32)
        # 运行期配置（由外部 setter 控制）
        self._use_topk = 0      # 0 表示不用 top-k；>0 表示 top-k 的 k
        self._use_gumbel = False
        #（缓存，供正则与日志使用）
        self.last_w_soft = None
        self.last_w_hard = None

    # 便捷 setter
    def set_tau(self, tau: float):
        self.tau.assign(float(tau))

    def set_topk(self, k: int | None):
        self._use_topk = int(k) if k is not None else 0

    def set_gumbel(self, flag: bool):
        self._use_gumbel = bool(flag)

    def call(self, x, training=None):
        # GAP -> MLP -> logits
        h  = self.fc1(self.gap(x), training=training)
        z  = self.fc2(h, training=training)      # [B,E] logits
        if training and self._use_gumbel:
            u = tf.random.uniform(tf.shape(z), 1e-6, 1-1e-6)
            g = -tf.math.log(-tf.math.log(u))
            z = z + g

        # 温度 softmax（软权重：用于统计/正则）
        w_soft = tf.nn.softmax(z / self.tau, axis=-1)  # [B,E]
        self.last_w_soft = w_soft

        # 混合权重：训练期可选硬 top-k（用于前向混合）
        w = w_soft
        if training and self._use_topk > 0:
            k = tf.minimum(self._use_topk, tf.shape(w_soft)[-1])
            vals, idx = tf.math.top_k(w_soft, k=k)
            mask = tf.reduce_sum(tf.one_hot(idx, depth=tf.shape(w_soft)[-1]), axis=1)  # [B,E]
            w_hard = tf.stop_gradient(mask) + w_soft - tf.stop_gradient(w_soft)
            w = w_hard / (tf.reduce_sum(w_hard, axis=-1, keepdims=True) + 1e-8)
            self.last_w_hard = w
        else:
            self.last_w_hard = None

        return w

# 3) MoA 残差块：E 个适配器 + 软路由 + 残差融合
class MoAResidualBlock(layers.Layer):
    """
    用法：
      y, w = MoAResidualBlock(C, num_experts=3, r_ratio=0.25, k=3)(x, training=True)
    """
    def __init__(self, channels, num_experts=3, r_ratio=0.25, k=3,
                 router_hidden=64, tau_init=1.5, name="moa_block"):
        super().__init__(name=name)
        self.adapters = [
            ResidualBottleneck(channels, k=k, r_ratio=r_ratio, name=f"{name}_adapter{i}")
            for i in range(num_experts)
        ]
        self.router = MoARouter(num_experts=num_experts, hidden=router_hidden, tau_init=tau_init, name=f"{name}_router")
        self.num_experts = num_experts

        # 便于训练时取出路由权重用于正则/可视化
        self.last_w = None
    
    def _rms_norm(self, y, eps=1e-6):
        # 按 (B,H,W,C) 的 RMS 归一到相近尺度，避免某个专家以幅值取胜
        rms = tf.sqrt(tf.reduce_mean(tf.square(y), axis=[1,2,3], keepdims=True) + eps)
        return y / rms

    def call(self, x, training=None):
        # 训练期用路由器内置的 topk/tau/gumbel 设置
        w = self.router(x, training=training)   # [B,E]
        # 统计/正则一律用软权重（避免硬化导致 H=0）
        self.last_w       = getattr(self.router, 'last_w_soft', w)

        outs = []
        for i, a in enumerate(self.adapters):
            y_i = a(x, training=training)                 # [B,H,W,C]
            y_i = self._rms_norm(y_i)                          # ★ 防幅值偏置
            w_i = tf.reshape(w[:, i], (-1, 1, 1, 1))      # [B,1,1,1]
            outs.append(w_i * y_i)

        y = x + (tf.add_n(outs) if len(outs) > 1 else outs[0])
        return y, w
