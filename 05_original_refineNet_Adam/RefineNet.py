#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 05:04:23 2025

@author: nephilim
"""
import os
import numpy as np
from matplotlib import pyplot, cm
import tensorflow as tf
from Mixture_of_Adapters import MoAResidualBlock
import training_func
import inference_ops


# %%
def conv(out_channels, kernel_size, bias=False, stride=1):
    return tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        use_bias=bias,
    )


# ------------------------------
# Spatial Attention (SALayer)
# ------------------------------
class SALayer(tf.keras.layers.Layer):
    def __init__(self, kernel_size=7, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=1, kernel_size=kernel_size, padding="same", use_bias=False
        )
        self.sigmoid = tf.keras.layers.Activation("sigmoid")

    def call(self, x, training=None):
        avg_out = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_out = tf.reduce_max(x, axis=-1, keepdims=True)
        y = tf.concat([avg_out, max_out], axis=-1)
        y = self.conv1(y)
        y = self.sigmoid(y)
        return x * y


# ------------------------------
# Channel Attention (CALayer)
# ------------------------------
class CALayer(tf.keras.layers.Layer):
    def __init__(self, channel, reduction=16, bias=False, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        mid = max(1, channel // reduction)
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.conv_du = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(mid, 1, padding="valid", use_bias=bias),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(channel, 1, padding="valid", use_bias=bias),
                tf.keras.layers.Activation("sigmoid"),
            ]
        )

    def call(self, x, training=None):
        y = self.avg_pool(x)  # [B,1,1,C]
        y = self.conv_du(y, training=training)
        return x * y


class Residual_CBAM_Block(tf.keras.layers.Layer):
    def __init__(
        self,
        cbam_feature,
        kernel_size,
        reduction=8,
        bias=False,
        act=None,
        sa_kernel=7,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.act = act if act is not None else tf.keras.layers.ReLU()
        self.body = tf.keras.Sequential(
            [
                conv(cbam_feature, kernel_size, bias=bias),
                self.act,
                conv(cbam_feature, kernel_size, bias=bias),
            ]
        )
        # CBAM: Channel Attention -> Spatial Attention
        self.ca = CALayer(
            channel=cbam_feature, reduction=reduction, bias=bias, name=f"{self.name}_ca"
        )
        self.sa = SALayer(kernel_size=sa_kernel, name=f"{self.name}_sa")

    def call(self, x, training=None):
        res = self.body(x, training=training)
        res = self.ca(res, training=training)
        res = self.sa(res, training=training)
        return x + res


def bilinear_resize_half():
    return tf.keras.layers.Lambda(
        lambda t: tf.image.resize(
            t, size=tf.cast(tf.shape(t)[1:3] // 2, tf.int32), method="bilinear"
        )
    )


def bilinear_resize_double():
    return tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")


class ResidualConvUnit(tf.keras.layers.Layer):
    def __init__(self, n_feat, kernel_size, bias, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.body = tf.keras.Sequential(
            [
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(
                    filters=n_feat,
                    kernel_size=kernel_size,
                    strides=1,
                    padding="same",
                    use_bias=bias,
                ),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(
                    filters=n_feat,
                    kernel_size=kernel_size,
                    strides=1,
                    padding="same",
                    use_bias=bias,
                ),
            ]
        )

    def call(self, x, training=None):
        y = self.body(x, training=training)
        return x + y


class DownSample(tf.keras.layers.Layer):
    def __init__(self, n_feat, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.seq = tf.keras.Sequential(
            [
                bilinear_resize_half(),
                tf.keras.layers.Conv2D(
                    filters=n_feat, kernel_size=1, padding="valid", use_bias=False
                ),
            ]
        )

    def call(self, x, training=None):
        return self.seq(x, training=training)


class UpSample(tf.keras.layers.Layer):
    def __init__(self, n_feat, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.seq = tf.keras.Sequential(
            [
                bilinear_resize_double(),
                tf.keras.layers.Conv2D(
                    filters=n_feat, kernel_size=1, padding="valid", use_bias=False
                ),
            ]
        )

    def call(self, x, training=None):
        return self.seq(x, training=training)


class DownSampleLevel(tf.keras.layers.Layer):
    def __init__(self, n_feat, level, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.seq = tf.keras.Sequential([DownSample(n_feat) for _ in range(level)])

    def call(self, x, training=None):
        return self.seq(x, training=training)


class UpSampleLevel(tf.keras.layers.Layer):
    def __init__(self, n_feat, level, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.seq = tf.keras.Sequential([UpSample(n_feat) for _ in range(level)])

    def call(self, x, training=None):
        return self.seq(x, training=training)


class DepthToSpaceLayer(tf.keras.layers.Layer):
    def __init__(self, scale, name=None, **kwargs):
        super(DepthToSpaceLayer, self).__init__(name=name, **kwargs)
        self.scale = scale

    def call(self, x, training=None):
        return tf.nn.depth_to_space(x, self.scale)

class UpSamplePixelShuffler(tf.keras.layers.Layer):
    def __init__(self, n_feat, scale, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.seq = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=(scale**2) * n_feat,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    use_bias=False,
                ),
                DepthToSpaceLayer(scale=scale),
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=3, 
                    padding='same', 
                    use_bias=False),
                # tf.keras.layers.BatchNormalization(),
                tf.keras.layers.PReLU(shared_axes=[1, 2]),
            ]
        )

    def call(self, x, training=None):
        return self.seq(x, training=training)


class ChainedBranch(tf.keras.layers.Layer):
    def __init__(self, n_feat, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.seq = tf.keras.Sequential(
            [
                tf.keras.layers.MaxPooling2D(
                    pool_size=(5, 5), strides=1, padding="same"
                ),
                tf.keras.layers.Conv2D(
                    filters=n_feat,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    use_bias=True,
                ),
            ]
        )

    def call(self, x, training=None):
        return self.seq(x, training=training)


class ChainedResidualPoolingNS(tf.keras.layers.Layer):
    def __init__(self, n_feat, level, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.relu = tf.keras.layers.ReLU()
        self.branches = [ChainedBranch(n_feat) for _ in range(level)]

    def call(self, x, training=None):
        y = self.relu(x)
        b = tf.identity(y)
        for br in self.branches:
            b = br(b, training=training)
            y = y + b
        return y


# %%
class RefineNet(tf.keras.Model):
    def __init__(self, n_feat, kernel_size, reduction, bias=True, **kwargs):
        super().__init__(name="backbone_refinenet", **kwargs)
        self.stage1 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=n_feat,
                    kernel_size=kernel_size,
                    strides=1,
                    padding="same",
                    use_bias=bias,
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.MaxPooling2D(),
            ],
            name="stage1",
        )
        self.cbam1 = Residual_CBAM_Block(n_feat, kernel_size, reduction, name="cbam1")
        self.stage2 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=n_feat * 2,
                    kernel_size=kernel_size,
                    strides=1,
                    padding="same",
                    use_bias=bias,
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.MaxPooling2D(),
            ],
            name="stage2",
        )
        self.cbam2 = Residual_CBAM_Block(
            n_feat * 2, kernel_size, reduction // 2, name="cbam2"
        )
        self.downsample1 = DownSample(n_feat * 2, name="downsample1")
        self.downsample2 = DownSampleLevel(n_feat * 2, 2, name="downsample2")
        self.downsample3 = DownSampleLevel(n_feat * 2, 3, name="downsample3")
        self.residual_conv_unit0 = tf.keras.Sequential(
            [
                ResidualConvUnit(n_feat * 2, kernel_size, bias=True),
                ResidualConvUnit(n_feat * 2, kernel_size, bias=True),
            ],
            name="residual_conv_unit0",
        )
        self.residual_conv_unit1 = tf.keras.Sequential(
            [
                ResidualConvUnit(n_feat * 2, kernel_size, bias=True),
                ResidualConvUnit(n_feat * 2, kernel_size, bias=True),
            ],
            name="residual_conv_unit1",
        )
        self.residual_conv_unit2 = tf.keras.Sequential(
            [
                ResidualConvUnit(n_feat * 2, kernel_size, bias=True),
                ResidualConvUnit(n_feat * 2, kernel_size, bias=True),
            ],
            name="residual_conv_unit2",
        )
        self.residual_conv_unit3 = tf.keras.Sequential(
            [
                ResidualConvUnit(n_feat * 2, kernel_size, bias=True),
                ResidualConvUnit(n_feat * 2, kernel_size, bias=True),
            ],
            name="residual_conv_unit3",
        )
        # self.upsample1=UpSample(n_feat*2, name='upsample1')
        # self.upsample2=UpSampleLevel(n_feat*2,2, name='upsample2')
        # self.upsample3=UpSampleLevel(n_feat*2,3, name='upsample3')
        self.upsample_pixel_shuffler1 = UpSamplePixelShuffler(
            n_feat * 2, 2, name="upsample_pixel_shuffler1"
        )
        self.upsample_pixel_shuffler2 = UpSamplePixelShuffler(
            n_feat * 2, 4, name="upsample_pixel_shuffler2"
        )
        self.upsample_pixel_shuffler3 = UpSamplePixelShuffler(
            n_feat * 2, 8, name="upsample_pixel_shuffler3"
        )
        self.chained_residual_pooling = ChainedResidualPoolingNS(
            n_feat * 2, 3, name="chained_residual_pooling"
        )
        self.residual_conv_unit = tf.keras.Sequential(
            [
                ResidualConvUnit(n_feat * 2, kernel_size, bias=True),
                ResidualConvUnit(n_feat * 2, kernel_size, bias=True),
            ],
            name="residual_conv_unit",
        )
        
        # self.output_stage1 = tf.keras.Sequential(
        #     [
        #         tf.keras.layers.Conv2DTranspose(
        #             filters=n_feat, kernel_size=kernel_size, strides=2, padding="same"
        #         ),
        #         tf.keras.layers.BatchNormalization(),
        #         tf.keras.layers.LeakyReLU(),
        #     ],
        #     name="output_stage1",
        # )

        self.output_stage1 = tf.keras.Sequential(
            [
                UpSample(n_feat),
                # tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
            ],
            name="output_stage1",
        )
        
        self.output_cbam1 = Residual_CBAM_Block(
            n_feat, kernel_size, reduction, name="output_cbam1"
        )
        
        # self.output_stage2 = tf.keras.Sequential(
        #     [
        #         tf.keras.layers.Conv2DTranspose(
        #             filters=n_feat // 2,
        #             kernel_size=kernel_size,
        #             strides=2,
        #             padding="same",
        #         ),
        #         tf.keras.layers.BatchNormalization(),
        #         tf.keras.layers.LeakyReLU(),
        #     ],
        #     name="output_stage2",
        # )
        
        self.output_stage2 = tf.keras.Sequential(
            [
                UpSample(n_feat // 2),
                # tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
            ],
            name="output_stage2",
        )
        
        self.output_cbam2 = Residual_CBAM_Block(
            n_feat // 2, kernel_size, reduction // 2, name="output_cbam2"
        )

    def call(self, x, training=None):
        # 256*256*1
        x = self.stage1(x, training=training)
        # 128*128*32
        x = self.cbam1(x, training=training)
        x = self.stage2(x, training=training)
        # 64*64*64
        x = self.cbam2(x, training=training)
        # 64*64*64
        x_1 = tf.identity(x)
        # 64*64*64
        # 64*64*64
        x_2 = self.downsample1(x, training=training)
        # 32*32*64
        # 64*64*64
        x_3 = self.downsample2(x, training=training)
        # 16*16*64
        # 64*64*64
        x_4 = self.downsample3(x, training=training)
        # 8*8*64
        x_1 = self.residual_conv_unit0(x_1, training=training)
        # 64*64*64
        x_2 = self.residual_conv_unit1(x_2, training=training)
        # 32*32*64
        x_3 = self.residual_conv_unit2(x_3, training=training)
        # 16*16*64
        x_4 = self.residual_conv_unit3(x_4, training=training)
        # 8*8*64
        # x_4=self.upsample3(x_4, training=training)
        x_4 = self.upsample_pixel_shuffler3(x_4, training=training)
        # 64*64*64
        # x_3=self.upsample2(x_3, training=training)
        x_3 = self.upsample_pixel_shuffler2(x_3, training=training)
        # 64*64*64
        # x_2=self.upsample1(x_2, training=training)
        x_2 = self.upsample_pixel_shuffler1(x_2, training=training)
        # 64*64*64
        x_1 = tf.identity(x_1)
        # 64*64*64
        x = x_1 + x_2 + x_3 + x_4
        x = self.chained_residual_pooling(x, training=training)
        x = self.residual_conv_unit(x, training=training)
        x = self.output_stage1(x, training=training)
        # 128*128*32
        x = self.output_cbam1(x, training=training)
        # 128*128*32
        x = self.output_stage2(x, training=training)
        # 256*256*16
        x = self.output_cbam2(x, training=training)
        return x


class RefineNetWithMoA(tf.keras.models.Model):
    def __init__(
        self,
        backbone: tf.keras.models.Model,
        feat_channels: int,
        num_experts=4,
        r_ratio=0.25,
        k=3,
        router_hidden=64,
    ):
        super().__init__(name="backbone_with_moa")
        self.backbone = backbone
        self.moa = MoAResidualBlock(
            feat_channels, num_experts, r_ratio, k, router_hidden, name="moa"
        )
        self.out_head = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=(1, 1), strides=(1, 1), padding="same"
                ),
                tf.keras.layers.Activation("tanh"),
            ],
            name="out_head",
        )
        self._last_moa_w = None

    def call(self, x, training=None):
        feat = self.backbone(x, training=training)
        feat_moa, w = self.moa(feat, training=training)
        self._last_moa_w = w
        y = self.out_head(feat_moa, training=training)
        return y


def freeze_all(layer):
    for sub in layer.submodules:
        sub.trainable = False


def unfreeze(layer):
    for sub in layer.submodules:
        sub.trainable = True


def freeze_bn(layer):
    for sub in layer.submodules:
        if isinstance(sub, tf.keras.layers.BatchNormalization):
            sub.trainable = False


def to_float(x):
    x = tf.convert_to_tensor(x)
    x = tf.reduce_mean(tf.reshape(x, [-1]))
    return float(x.numpy())


def to_list(x):
    x = tf.convert_to_tensor(x)
    return [float(v) for v in tf.reshape(x, [-1]).numpy()]


def log_router_weights_from_tensor(w_tensor, stage: str, epoch: int, step_idx: int):
    if w_tensor is None:
        return
    if tf.size(w_tensor) == 0:
        return
    w_np = w_tensor.numpy() if hasattr(w_tensor, "numpy") else np.array(w_tensor)
    out_path = os.path.join(
        ROUTER_LOG_DIR, f"router_w_{stage:s}_e{epoch:03d}_b{step_idx:04d}.npy"
    )
    np.save(out_path, w_np.astype(np.float32))


def training_backbone(
    model,
    training_dataset,
    validation_dataset,
    BATCH_SIZE,
    epochs_backbone,
    optimizer_backbone,
    test_input,
):
    loss_history_backbone = []
    validation_loss_history_backbone = []
    # %%
    # ---------------- STAGE 0: pre-train backbone + out_head; freeze MoA ----------------
    unfreeze(model)
    for key in [
        "moa",
    ]:
        try:
            model.get_layer(key).trainable = False
        except ValueError:
            pass
    # freeze_bn(model)

    loss_epoch_history = []
    validation_loss_epoch_history = []
    for epoch in range(epochs_backbone):
        for step_, (x, y) in enumerate(training_dataset):
            metrics = training_func.compute_apply_gradients_backbone(
                model, x, y, optimizer_backbone
            )
            loss_epoch_history.append(metrics["loss"].numpy())
            if step_ % 1 == 0:
                print(
                    f"[Stage0][Ep {epoch}]:\n"
                    f"step {step_} loss={to_float(metrics['loss']):.4e},\n"
                    f"loss_charb={to_float(metrics['loss_charb']):.4e},\n"
                    f"loss_ps={to_float(metrics['loss_ps']):.4e}"
                )
        if epoch % 10 == 0:
            for step_, (vx, vy) in enumerate(validation_dataset):
                val = training_func.compute_backbone(model, vx, vy)
                validation_loss_epoch_history.append(val["loss"].numpy())
                if step_ % 1 == 0:
                    print(
                        f"[Stage0][Val {epoch}]:\n"
                        f"step {step_} loss={to_float(val['loss']):.4e},\n"
                        f"loss_charb={to_float(val['loss_charb']):.4e},\n"
                        f"loss_ps={to_float(val['loss_ps']):.4e}\n"
                    )
        if epoch % 10 == 0:
            clutter_inference, clutter_prediction = inference_ops.inference_patch(
                model, test_input["clutter"], resize_shape=(256, 512)
            )
            crts_inference, crts_prediction = inference_ops.inference_patch(
                model, test_input["crts"], resize_shape=(256, 512)
            )
            rebar1_inference, rebar1_prediction = inference_ops.inference_patch(
                model, test_input["rebar1"], resize_shape=(256, 512)
            )
            rebar2_inference, rebar2_prediction = inference_ops.inference_patch(
                model, test_input["rebar2"], resize_shape=(256, 512)
            )

            pyplot.figure(1)
            pyplot.subplot(221)
            pyplot.imshow(clutter_inference)
            pyplot.subplot(222)
            pyplot.imshow(crts_inference)
            pyplot.subplot(223)
            pyplot.imshow(rebar1_inference)
            pyplot.subplot(224)
            pyplot.imshow(rebar2_inference)
            pyplot.savefig("./Results/original.png")

            pyplot.figure(2)
            pyplot.subplot(221)
            pyplot.imshow(clutter_prediction)
            pyplot.subplot(222)
            pyplot.imshow(crts_prediction)
            pyplot.subplot(223)
            pyplot.imshow(rebar1_prediction)
            pyplot.subplot(224)
            pyplot.imshow(rebar2_prediction)
            pyplot.savefig("./Results/backnone_prediction_%s.png" % epoch)

            model.save("./Weights/backbone_model_%s" % epoch, save_format="tf")

        loss_history_backbone.append(np.mean(loss_epoch_history))
        validation_loss_history_backbone.append(np.mean(validation_loss_epoch_history))
    return {
        "model": model,
        "loss_history_backbone": loss_history_backbone,
        "validation_loss_history_backbone": validation_loss_history_backbone,
    }


def training_moa(
    model,
    training_dataset,
    validation_dataset,
    BATCH_SIZE,
    epochs_moa,
    optimizer_moa,
    test_input,
):
    loss_rec_history_moa = []
    validation_rec_history_moa = []
    loss_history_moa = []
    validation_loss_history_moa = []
    # %%
    # ---------------- STAGE 1: freeze backbone + out_head; train MoA only ----------------
    freeze_all(model)
    for key in [
        "moa",
    ]:
        try:
            model.get_layer(key).trainable = True
        except ValueError:
            pass
    # freeze_bn(model)

    class EMA:
        def __init__(self, alpha=0.9):
            self.alpha = alpha
            self.v = None

        def update(self, x):
            x = float(x)
            if self.v is None:
                self.v = x
            else:
                self.v = self.alpha * self.v + (1 - self.alpha) * x
            return self.v

        @property
        def value(self):
            return self.v if self.v is not None else 0.0

    ema_H = EMA(alpha=0.9)  # EMA of H_bar
    ema_min = EMA(alpha=0.9)  # EMA of the minimum load among w_mean

    entered_phase_c = False  # one-time perturbation flag when entering phase C
    warmup_countdown = 0  # warm-up window counter (in steps)
    WARMUP_STEPS = 300  # warm-up duration steps
    MIN_LOAD_TH = 0.10  # guard threshold (enabled while t < 0.8)

    loss_epoch_history = []
    validation_loss_epoch_history = []
    loss_rec_epoch_history = []
    validation_rec_epoch_history = []
    T = float(epochs_moa)
    E = tf.cast(getattr(model.moa, "num_experts", 4), tf.float32)
    for epoch in range(epochs_moa):
        t = epoch / max(1.0, T)  # ∈[0,1)
        if t < 0.4:
            # Phase A: pure soft routing, exploration; avoid early collapse
            model.moa.router.set_topk(0)
            model.moa.router.set_gumbel(True)
            tau = 1.5 - 1.0 * (t / 0.4)  # 2.0 → 1.0
            lambda_sparse = 0.8
            lambda_div = 1.0

        elif t < 0.8:
            # Phase B: soft + hard combined; start differentiating but avoid over-specializing
            model.moa.router.set_topk(2)
            model.moa.router.set_gumbel(True)
            tau = 1.0 - 0.2 * ((t - 0.4) / 0.4)  # 1.0 → 0.8
            lambda_sparse = 0.8
            lambda_div = 1.0

        else:
            # Phase C: convergence / deployment oriented; sparser and more specialized
            model.moa.router.set_topk(1)
            model.moa.router.set_gumbel(True)
            tau = 0.7 - 0.2 * ((t - 0.8) / 0.2)  # 0.8 → 0.6
            lambda_sparse = 0.8  # slightly stronger force
            lambda_div = 1.0

        if (0.6 <= t < 0.8) and ema_H.value:
            if ema_H.value < 0.60:
                model.moa.router.set_topk(1)
                model.moa.router.set_gumbel(True)
                tau = max(0.55, tau - 0.1)
                lambda_sparse = max(1.2, float(lambda_sparse))
                lambda_div = min(0.1, float(lambda_div))
        if (t >= 0.8) and (not entered_phase_c):
            try:
                E = int(getattr(model.moa.router, "fc2").units)  # number of experts
                model.moa.router.fc2.bias.assign_add(1e-3 * tf.random.normal([E]))
            except Exception:
                pass
            entered_phase_c = True

        model.moa.router.set_tau(float(tau))

        for step_, (x, y) in enumerate(training_dataset):
            metrics = training_func.compute_apply_gradients_moa(
                model,
                x,
                y,
                optimizer_moa,
                lambda_sparse=lambda_sparse,
                lambda_diversity=lambda_div,
            )

            H_bar = float(metrics.get("H_bar", 0.0))
            w_mean = metrics.get("w_mean", None)

            ema_H.update(H_bar)
            if w_mean is not None:
                try:
                    min_load = float(np.min(w_mean))
                except Exception:
                    min_load = float(tf.reduce_min(w_mean))
                ema_min.update(min_load)
            else:
                min_load = 0.0

            # ===== warm-up guard: if t < 0.8 and min-load is too low, briefly soften =====
            if (t < 0.8) and (ema_min.value < MIN_LOAD_TH) and (warmup_countdown == 0):
                warmup_countdown = WARMUP_STEPS

            if warmup_countdown > 0:
                # full warm-up: disable top-k and increase temperature
                model.moa.router.set_topk(0)
                model.moa.router.set_tau(float(tau + 0.2))
                warmup_countdown -= 1

            log_router_weights_from_tensor(
                metrics.get("w", None), "stage1", epoch, step_
            )
            loss_epoch_history.append(metrics["loss"].numpy())
            loss_rec_epoch_history.append(metrics["loss_rec"].numpy())
            if step_ % 1 == 0:
                print(
                    f"[Stage1][Ep {epoch}]:\n"
                    f"loss={to_float(metrics['loss']):.4e},\n"
                    f"loss_rec={to_float(metrics['loss_rec']):.4e},\n"
                    f"loss_sparse={to_float(metrics['loss_sparse']):.4e},\n"
                    f"loss_diversity={to_float(metrics['loss_diversity']):.4e},\n"
                    f"w_mean={to_list(metrics['w_mean'])},\n"
                    f"top1={to_float(metrics['top1'])},\n"
                    f"tau={float(tau):.2f},\n"
                    f"lam=(sp:{lambda_sparse:+.3f}, dv:{lambda_div:.3f}),\n"
                    f"H_bar={to_float(metrics['H_bar']):.3f}\n"
                )

        if epoch % 10 == 0:
            for step_, (vx, vy) in enumerate(validation_dataset):
                val = training_func.compute_moa(
                    model,
                    vx,
                    vy,
                    lambda_sparse=lambda_sparse,
                    lambda_diversity=lambda_div,
                )
                validation_loss_epoch_history.append(val["loss"].numpy())
                validation_rec_epoch_history.append(val["loss_rec"].numpy())
                if step_ % 1 == 0:
                    print(
                        f"[Stage1][Val {epoch}]:\n"
                        f"loss={to_float(metrics['loss']):.4e},\n"
                        f"loss_rec={to_float(metrics['loss_rec']):.4e},\n"
                        f"loss_sparse={to_float(metrics['loss_sparse']):.4e},\n"
                        f"loss_diversity={to_float(metrics['loss_diversity']):.4e},\n"
                        f"w_mean={to_list(metrics['w_mean'])},\n"
                        f"top1={to_float(metrics['top1'])},\n"
                        f"tau={float(tau):.2f},\n"
                        f"lam=(sp:{lambda_sparse:+.3f}, dv:{lambda_div:.3f}),\n"
                        f"H_bar={to_float(metrics['H_bar']):.3f}\n"
                    )

        w_mean = tf.reduce_mean(metrics["w"], axis=0) if "w" in metrics else None
        if w_mean is not None:
            min_load = float(tf.reduce_min(w_mean))
            # if severe imbalance occurs (one expert almost never used), apply a temporary warm-up round
            if min_load < 0.05 and t < 0.8:
                print("[Guard] severe imbalance -> temporary soften")
                model.moa.router.set_topk(0)
                model.moa.router.set_tau(min(float(model.moa.router.tau) + 0.2, 2.0))

        if epoch % 10 == 0:
            clutter_inference, clutter_prediction = inference_ops.inference_patch(
                model, test_input["clutter"], resize_shape=(256, 512)
            )
            crts_inference, crts_prediction = inference_ops.inference_patch(
                model, test_input["crts"], resize_shape=(256, 512)
            )
            rebar1_inference, rebar1_prediction = inference_ops.inference_patch(
                model, test_input["rebar1"], resize_shape=(256, 512)
            )
            rebar2_inference, rebar2_prediction = inference_ops.inference_patch(
                model, test_input["rebar2"], resize_shape=(256, 512)
            )

            pyplot.figure(1)
            pyplot.subplot(221)
            pyplot.imshow(clutter_inference)
            pyplot.subplot(222)
            pyplot.imshow(crts_inference)
            pyplot.subplot(223)
            pyplot.imshow(rebar1_inference)
            pyplot.subplot(224)
            pyplot.imshow(rebar2_inference)
            pyplot.savefig("./Results/original.png")

            pyplot.figure(2)
            pyplot.subplot(221)
            pyplot.imshow(clutter_prediction)
            pyplot.subplot(222)
            pyplot.imshow(crts_prediction)
            pyplot.subplot(223)
            pyplot.imshow(rebar1_prediction)
            pyplot.subplot(224)
            pyplot.imshow(rebar2_prediction)
            pyplot.savefig("./Results/moa_prediction_%s.png" % epoch)

            model.save("./Weights/moa_model_%s" % epoch, save_format="tf")

        loss_history_moa.append(np.mean(loss_epoch_history))
        validation_loss_history_moa.append(np.mean(validation_loss_epoch_history))
        loss_rec_history_moa.append(np.mean(loss_rec_epoch_history))
        validation_rec_history_moa.append(np.mean(validation_rec_epoch_history))
    return {
        "model": model,
        "loss_history_moa": loss_history_moa,
        "validation_loss_history_moa": validation_loss_history_moa,
        "loss_rec_history_moa": loss_rec_history_moa,
        "validation_rec_history_moa": validation_rec_history_moa,
    }


def training_moa_part_backbone(
    model,
    training_dataset,
    validation_dataset,
    BATCH_SIZE,
    epochs_moa_part_backbone,
    optimizer_moa_part_backbone_moa,
    optimizer_moa_part_backbone_top,
    test_input,
):

    loss_history_moa_part_backbone = []
    validation_loss_history_moa_part_backbone = []
    loss_rec_history_moa_part_backbone = []
    validation_rec_history_moa_part_backbone = []
    # %%
    # ---------------- STAGE 1: freeze backbone + out_head; train MoA only ----------------
    freeze_all(model.backbone)
    # only enable a few top layers
    for key in ["output_stage1", "output_cbam1", "output_stage2", "output_cbam2"]:
        try:
            model.backbone.get_layer(key).trainable = True
        except ValueError:
            pass
    model.out_head.trainable = True
    unfreeze(model.moa)
    freeze_bn(model)

    class EMA:
        def __init__(self, alpha=0.9):
            self.alpha = alpha
            self.v = None

        def update(self, x):
            x = float(x)
            if self.v is None:
                self.v = x
            else:
                self.v = self.alpha * self.v + (1 - self.alpha) * x
            return self.v

        @property
        def value(self):
            return self.v if self.v is not None else 0.0

    ema_H = EMA(alpha=0.9)  # EMA of H_bar
    ema_min = EMA(alpha=0.9)  # EMA of the minimum load among w_mean

    entered_phase_c = False  # one-time perturbation flag when entering phase C
    warmup_countdown = 0  # warm-up window counter (in steps)
    WARMUP_STEPS = 300  # warm-up duration steps
    MIN_LOAD_TH = 0.10  # guard threshold (enabled while t < 0.8)

    loss_epoch_history = []
    validation_loss_epoch_history = []
    loss_rec_epoch_history = []
    validation_rec_epoch_history = []
    T = float(epochs_moa_part_backbone)
    E = tf.cast(getattr(model.moa, "num_experts", 4), tf.float32)
    for epoch in range(epochs_moa_part_backbone):
        t = epoch / max(1.0, T)  # ∈[0,1)
        if t < 0.4:
            # Phase A: pure soft routing, exploration; avoid early collapse
            model.moa.router.set_topk(0)
            model.moa.router.set_gumbel(True)
            tau = 1.5 - 1.0 * (t / 0.4)  # 2.0 → 1.0
            lambda_sparse = 0.8
            lambda_div = 1.0

        elif t < 0.8:
            # Phase B: soft + hard combined; start differentiating but avoid over-specializing
            model.moa.router.set_topk(2)
            model.moa.router.set_gumbel(True)
            tau = 1.0 - 0.2 * ((t - 0.4) / 0.4)  # 1.0 → 0.8
            lambda_sparse = 0.8
            lambda_div = 1.0

        else:
            # Phase C: convergence / deployment oriented; sparser and more specialized
            model.moa.router.set_topk(1)
            model.moa.router.set_gumbel(True)
            tau = 0.7 - 0.2 * ((t - 0.8) / 0.2)  # 0.8 → 0.6
            lambda_sparse = 0.8  # slightly stronger force
            lambda_div = 1.0

        if (0.6 <= t < 0.8) and ema_H.value:
            if ema_H.value < 0.60:
                model.moa.router.set_topk(1)
                model.moa.router.set_gumbel(True)
                tau = max(0.55, tau - 0.1)
                lambda_sparse = max(1.2, float(lambda_sparse))
                lambda_div = min(0.1, float(lambda_div))
        if (t >= 0.8) and (not entered_phase_c):
            try:
                E = int(getattr(model.moa.router, "fc2").units)  # number of experts
                model.moa.router.fc2.bias.assign_add(1e-3 * tf.random.normal([E]))
            except Exception:
                pass
            entered_phase_c = True

        model.moa.router.set_tau(float(tau))

        for step_, (x, y) in enumerate(training_dataset):
            metrics = training_func.compute_apply_gradients_moa_part_backbone(
                model,
                x,
                y,
                optimizer_moa_part_backbone_moa,
                optimizer_moa_part_backbone_top,
                lambda_sparse=lambda_sparse,
                lambda_diversity=lambda_div,
            )

            H_bar = float(metrics.get("H_bar", 0.0))
            w_mean = metrics.get("w_mean", None)

            ema_H.update(H_bar)
            if w_mean is not None:
                try:
                    min_load = float(np.min(w_mean))
                except Exception:
                    min_load = float(tf.reduce_min(w_mean))
                ema_min.update(min_load)
            else:
                min_load = 0.0

            # ===== warm-up guard: if t < 0.8 and min-load is too low, briefly soften =====
            if (t < 0.8) and (ema_min.value < MIN_LOAD_TH) and (warmup_countdown == 0):
                warmup_countdown = WARMUP_STEPS

            if warmup_countdown > 0:
                # full warm-up: disable top-k and increase temperature
                model.moa.router.set_topk(0)
                model.moa.router.set_tau(float(tau + 0.2))
                warmup_countdown -= 1

            log_router_weights_from_tensor(
                metrics.get("w", None), "stage1", epoch, step_
            )
            loss_epoch_history.append(metrics["loss"].numpy())
            loss_rec_epoch_history.append(metrics["loss_rec"].numpy())
            if step_ % 1 == 0:
                print(
                    f"[Stage2][Ep {epoch}]:\n"
                    f"loss={to_float(metrics['loss']):.4e},\n"
                    f"loss_rec={to_float(metrics['loss_rec']):.4e},\n"
                    f"loss_sparse={to_float(metrics['loss_sparse']):.4e},\n"
                    f"loss_diversity={to_float(metrics['loss_diversity']):.4e},\n"
                    f"w_mean={to_list(metrics['w_mean'])},\n"
                    f"top1={to_float(metrics['top1'])},\n"
                    f"tau={float(tau):.2f},\n"
                    f"lam=(sp:{lambda_sparse:+.3f}, dv:{lambda_div:.3f}),\n"
                    f"H_bar={to_float(metrics['H_bar']):.3f}\n"
                )

        if epoch % 10 == 0:
            for step_, (vx, vy) in enumerate(validation_dataset):
                val = training_func.compute_moa_part_backbone(
                    model,
                    vx,
                    vy,
                    lambda_sparse=lambda_sparse,
                    lambda_diversity=lambda_div,
                )
                validation_loss_epoch_history.append(val["loss"].numpy())
                validation_rec_epoch_history.append(val["loss_rec"].numpy())
                if step_ % 1 == 0:
                    print(
                        f"[Stage2][Val {epoch}]:\n"
                        f"loss={to_float(metrics['loss']):.4e},\n"
                        f"loss_rec={to_float(metrics['loss_rec']):.4e},\n"
                        f"loss_sparse={to_float(metrics['loss_sparse']):.4e},\n"
                        f"loss_diversity={to_float(metrics['loss_diversity']):.4e},\n"
                        f"w_mean={to_list(metrics['w_mean'])},\n"
                        f"top1={to_float(metrics['top1'])},\n"
                        f"tau={float(tau):.2f},\n"
                        f"lam=(sp:{lambda_sparse:+.3f}, dv:{lambda_div:.3f}),\n"
                        f"H_bar={to_float(metrics['H_bar']):.3f}\n"
                    )

        w_mean = tf.reduce_mean(metrics["w"], axis=0) if "w" in metrics else None
        if w_mean is not None:
            min_load = float(tf.reduce_min(w_mean))
            # if severe imbalance occurs (one expert almost never used), apply a temporary warm-up round
            if min_load < 0.05 and t < 0.8:
                print("[Guard] severe imbalance -> temporary soften")
                model.moa.router.set_topk(0)
                model.moa.router.set_tau(min(float(model.moa.router.tau) + 0.2, 2.0))

        if epoch % 10 == 0:
            clutter_inference, clutter_prediction = inference_ops.inference_patch(
                model, test_input["clutter"], resize_shape=(256, 512)
            )
            crts_inference, crts_prediction = inference_ops.inference_patch(
                model, test_input["crts"], resize_shape=(256, 512)
            )
            rebar1_inference, rebar1_prediction = inference_ops.inference_patch(
                model, test_input["rebar1"], resize_shape=(256, 512)
            )
            rebar2_inference, rebar2_prediction = inference_ops.inference_patch(
                model, test_input["rebar2"], resize_shape=(256, 512)
            )

            pyplot.figure(1)
            pyplot.subplot(221)
            pyplot.imshow(clutter_inference)
            pyplot.subplot(222)
            pyplot.imshow(crts_inference)
            pyplot.subplot(223)
            pyplot.imshow(rebar1_inference)
            pyplot.subplot(224)
            pyplot.imshow(rebar2_inference)
            pyplot.savefig("./Results/original.png")

            pyplot.figure(2)
            pyplot.subplot(221)
            pyplot.imshow(clutter_prediction)
            pyplot.subplot(222)
            pyplot.imshow(crts_prediction)
            pyplot.subplot(223)
            pyplot.imshow(rebar1_prediction)
            pyplot.subplot(224)
            pyplot.imshow(rebar2_prediction)
            pyplot.savefig("./Results/moa_part_backbone_prediction_%s.png" % epoch)

            model.save("./Weights/moa_part_backbone_model_%s" % epoch, save_format="tf")

        loss_history_moa_part_backbone.append(np.mean(loss_epoch_history))
        validation_loss_history_moa_part_backbone.append(
            np.mean(validation_loss_epoch_history)
        )
        loss_rec_history_moa_part_backbone.append(np.mean(loss_rec_epoch_history))
        validation_rec_history_moa_part_backbone.append(
            np.mean(validation_rec_epoch_history)
        )
    return {
        "model": model,
        "loss_history_moa_part_backbone": loss_history_moa_part_backbone,
        "validation_loss_history_moa_part_backbone": validation_loss_history_moa_part_backbone,
        "loss_rec_history_moa_part_backbone": loss_rec_history_moa_part_backbone,
        "validation_rec_history_moa_part_backbone": validation_rec_history_moa_part_backbone,
    }


def print_layer_tree(layer, depth=0, max_depth=99):
    indent = "  " * depth
    print(f"{indent}{layer.name:40s}  trainable={layer.trainable}")
    if depth >= max_depth:
        return
    if hasattr(layer, "layers"):
        for sub in layer.layers:
            print_layer_tree(sub, depth + 1, max_depth)


def cosine_lr(initial_lr, steps_per_epoch, epochs, final_ratio=0.1):
    """
    CosineDecay learning rate: cosine-decays from initial_lr down to initial_lr * final_ratio
    final_ratio = 0.1 means it converges to 10% of the initial value at the end
    """
    decay_steps = int(steps_per_epoch * epochs)
    return tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        alpha=final_ratio,  # terminal = initial_lr * alpha
    )


def exponential_lr(initial_lr, steps_per_epoch, epochs, decay_rate=0.9):
    decay_steps = int(steps_per_epoch * epochs)
    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr, decay_steps=decay_steps, decay_rate=decay_rate
    )


if __name__ == "__main__":
    ROUTER_LOG_DIR = "./router_w_logs"
    os.makedirs(ROUTER_LOG_DIR, exist_ok=True)
    # %%
    backbone = RefineNet(n_feat=32, kernel_size=3, reduction=8)
    moa_model = RefineNetWithMoA(
        backbone, feat_channels=16, num_experts=4, r_ratio=0.25, k=3, router_hidden=64
    )
    # %%
    x = tf.keras.Input(shape=(256, 256, 1))
    _ = moa_model(x)
    moa_model.summary()
    moa_model.backbone.summary()
    # %%
    print_layer_tree(moa_model)
    print("----- BACKBONE ONLY -----")
    print_layer_tree(moa_model.backbone)
    # %%
    # for l in moa_model.backbone.layers:
    #     print(l.name, l.trainable)

    # for l in moa_model.layers:
    #     print(l.name, l.trainable)
    # #%%
    # for sub in moa_model.backbone.submodules:
    #     sub.trainable = False
    # print_layer_tree(moa_model)
    # print("----- BACKBONE ONLY -----")
    # print_layer_tree(moa_model.backbone)
    # #%%
    # for key in ["output_stage1", "output_cbam1", "output_stage2", "output_cbam2"]:
    #     moa_model.backbone.get_layer(key).trainable = True
    # for sub in moa_model.moa.submodules:
    #     sub.trainable = True
    # print_layer_tree(moa_model)
    # print("----- BACKBONE ONLY -----")
    # print_layer_tree(moa_model.backbone)
    # #%%
    # for sub in moa_model.submodules:
    #     if isinstance(sub, tf.keras.layers.BatchNormalization):
    #         sub.trainable = False
    # %%
    train_input = np.load("data_tanh_input.npy").astype("float32")
    train_output = np.load("data_tanh_output.npy").astype("float32")
    validation_input = train_input[::10]
    validation_output = train_output[::10]

    TOTAL_SAMPLES = 6567
    BATCH_SIZE = 16
    TRAIN_BUF = TOTAL_SAMPLES * 2
    training_dataset = (
        tf.data.Dataset.from_tensor_slices((train_input, train_output))
        .shuffle(TRAIN_BUF)
        .batch(BATCH_SIZE)
    )
    validation_dataset = (
        tf.data.Dataset.from_tensor_slices((validation_input, validation_output))
        .shuffle(TRAIN_BUF)
        .batch(BATCH_SIZE)
    )
    # %%
    epochs_backbone = 101
    epochs_moa = 51
    epochs_moa_part_backbone = 51
    steps_per_epoch = int(tf.math.ceil(TOTAL_SAMPLES / BATCH_SIZE))
    
    lr_backbone = exponential_lr(
        initial_lr=2e-4, steps_per_epoch=steps_per_epoch, epochs=5, decay_rate=0.9
        )
    
    # lr_backbone = cosine_lr(
    #     initial_lr=2e-4, steps_per_epoch=steps_per_epoch, epochs=5, final_ratio=0.1
    # )
    
    optimizer_backbone = tf.keras.optimizers.Adam(
        learning_rate=lr_backbone
    )

    lr_moa = exponential_lr(
        initial_lr=1e-4, steps_per_epoch=steps_per_epoch, epochs=5, decay_rate=0.9
        )
    
    # lr_moa = cosine_lr(
    #     initial_lr=1e-4, steps_per_epoch=steps_per_epoch, epochs=5, final_ratio=0.2
    # )
    optimizer_moa = tf.keras.optimizers.Adam(learning_rate=lr_moa)
    
    lr_moa_part_backbone_top = exponential_lr(
        initial_lr=5e-5, steps_per_epoch=steps_per_epoch, epochs=5, decay_rate=0.9
        )
    lr_moa_part_backbone_moa = exponential_lr(
        initial_lr=1e-4, steps_per_epoch=steps_per_epoch, epochs=5, decay_rate=0.9
        )
    
    # lr_moa_part_backbone_top = cosine_lr(
    #     initial_lr=5e-5, steps_per_epoch=steps_per_epoch, epochs=5, final_ratio=0.25
    # )
    # lr_moa_part_backbone_moa = cosine_lr(
    #     initial_lr=1e-4, steps_per_epoch=steps_per_epoch, epochs=5, final_ratio=0.2
    # )
    
    optimizer_moa_part_backbone_top = tf.keras.optimizers.Adam(
        learning_rate=lr_moa_part_backbone_top
    )
    
    optimizer_moa_part_backbone_moa = tf.keras.optimizers.Adam(
        learning_rate=lr_moa_part_backbone_moa
    )

    test_input_clutter = np.load("00 clutter_testing.npy").astype("float32")
    test_input_crts = np.load("01 crts_testing.npy").astype("float32")
    test_input_rebar1 = np.load("02 rebar_testing.npy").astype("float32")
    test_input_rebar2 = np.load("03 rebar_testing.npy").astype("float32")
    test_input = {
        "clutter": test_input_clutter,
        "crts": test_input_crts,
        "rebar1": test_input_rebar1,
        "rebar2": test_input_rebar2,
    }

    model_info_backbone = training_backbone(
        moa_model,
        training_dataset,
        validation_dataset,
        BATCH_SIZE,
        epochs_backbone,
        optimizer_backbone,
        test_input,
    )
    np.save("loss_history_backbone.npy", model_info_backbone["loss_history_backbone"])
    np.save(
        "validation_loss_history_backbone.npy",
        model_info_backbone["validation_loss_history_backbone"],
    )

    model_info_moa = training_moa(
        model_info_backbone["model"],
        training_dataset,
        validation_dataset,
        BATCH_SIZE,
        epochs_moa,
        optimizer_moa,
        test_input,
    )
    np.save("loss_history_moa.npy", model_info_moa["loss_history_moa"])
    np.save(
        "validation_loss_history_moa.npy", model_info_moa["validation_loss_history_moa"]
    )
    np.save("loss_rec_history_moa.npy", model_info_moa["loss_rec_history_moa"])
    np.save(
        "validation_rec_history_moa.npy", model_info_moa["validation_rec_history_moa"]
    )

    model_info_moa_part_backbone = training_moa_part_backbone(
        model_info_moa["model"],
        training_dataset,
        validation_dataset,
        BATCH_SIZE,
        epochs_moa_part_backbone,
        optimizer_moa_part_backbone_moa,
        optimizer_moa_part_backbone_top,
        test_input,
    )
    np.save(
        "loss_history_moa_part_backbone.npy",
        model_info_moa_part_backbone["loss_history_moa_part_backbone"],
    )
    np.save(
        "validation_loss_history_moa_part_backbone.npy",
        model_info_moa_part_backbone["validation_loss_history_moa_part_backbone"],
    )
    np.save(
        "loss_rec_history_moa_part_backbone.npy",
        model_info_moa_part_backbone["loss_rec_history_moa_part_backbone"],
    )
    np.save(
        "validation_rec_history_moa_part_backbone.npy",
        model_info_moa_part_backbone["validation_rec_history_moa_part_backbone"],
    )
