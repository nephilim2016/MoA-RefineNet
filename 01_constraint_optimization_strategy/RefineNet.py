#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 13:38:09 2022

@author: gzsz
"""
import os
import numpy as np
from matplotlib import pyplot,cm
import tensorflow as tf
import skimage.transform
import Normalization
from Mixture_of_Adapters import MoAResidualBlock

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

def conv(out_channels, kernel_size, bias=False, stride=1):
    return tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=kernel_size,
        strides=stride,
        padding='same',
        use_bias=bias
    )

# ------------------------------
# Spatial Attention (SALayer)
# ------------------------------
class SALayer(tf.keras.layers.Layer):
    def __init__(self, kernel_size=7, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=kernel_size,
            padding='same',
            use_bias=False
        )
        self.sigmoid = tf.keras.layers.Activation('sigmoid')
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
        self.conv_du = tf.keras.Sequential([
            tf.keras.layers.Conv2D(mid, 1, padding='valid', use_bias=bias),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(channel, 1, padding='valid', use_bias=bias),
            tf.keras.layers.Activation('sigmoid')
        ])
    def call(self, x, training=None):
        y = self.avg_pool(x)                 # [B,1,1,C]
        y = self.conv_du(y, training=training)
        return x * y


class Residual_CBAM_Block(tf.keras.layers.Layer):
    def __init__(self, cbam_feature, kernel_size, reduction=8, bias=False, act=None, sa_kernel=7, name=None, **kwargs):
        super().__init__(name=name,**kwargs)
        self.act = act if act is not None else tf.keras.layers.ReLU()
        self.body = tf.keras.Sequential([
            conv(cbam_feature, kernel_size, bias=bias),
            self.act,
            conv(cbam_feature, kernel_size, bias=bias),
        ])
        # CBAM: Channel Attention -> Spatial Attention
        self.ca = CALayer(channel=cbam_feature, reduction=reduction, bias=bias, name=f"{self.name}_ca")
        self.sa = SALayer(kernel_size=sa_kernel, name=f"{self.name}_sa")
    def call(self, x, training=None):
        res = self.body(x, training=training)     
        res = self.ca(res, training=training)     
        res = self.sa(res, training=training)
        return x + res
    
def bilinear_resize_half():
    return tf.keras.layers.Lambda(lambda t: tf.image.resize(
        t, size=tf.cast(tf.shape(t)[1:3] // 2, tf.int32), method='bilinear'
    ))

def bilinear_resize_double():
    return tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')

class ResidualConvUnit(tf.keras.layers.Layer):
    def __init__(self, n_feat, kernel_size, bias, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.body = tf.keras.Sequential([
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=n_feat, kernel_size=kernel_size, strides=1, padding='same', use_bias=bias),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=n_feat, kernel_size=kernel_size, strides=1, padding='same', use_bias=bias)
            ])
    def call(self,x,training=None):
        y=self.body(x,training=training)
        return x+y

class DownSample(tf.keras.layers.Layer):
    def __init__(self, n_feat, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.seq = tf.keras.Sequential([
            bilinear_resize_half(),
            tf.keras.layers.Conv2D(filters=n_feat, kernel_size=1, padding='valid', use_bias=False)
        ])
    def call(self, x, training=None):
        return self.seq(x, training=training)

class UpSample(tf.keras.layers.Layer):
    def __init__(self, n_feat, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.seq = tf.keras.Sequential([
            bilinear_resize_double(),
            tf.keras.layers.Conv2D(filters=n_feat, kernel_size=1, padding='valid', use_bias=False)
        ])
    def call(self, x, training=None):
        return self.seq(x, training=training)
    
class DownSampleLevel(tf.keras.layers.Layer):
    def __init__(self, n_feat, level, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.seq = tf.keras.Sequential(
            [DownSample(n_feat) for _ in range(level)]
            )
    def call(self, x, training=None):
        return self.seq(x, training=training)
    
class UpSampleLevel(tf.keras.layers.Layer):
    def __init__(self, n_feat, level, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.seq = tf.keras.Sequential(
            [UpSample(n_feat) for _ in range(level)]
            )
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
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=(scale**2) * n_feat, kernel_size=3, strides=1, padding='same', use_bias=False
            ),
            DepthToSpaceLayer(scale=scale),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.PReLU(shared_axes=[1, 2])
        ])
    def call(self, x, training=None):
        return self.seq(x, training=training)
    
class ChainedBranch(tf.keras.layers.Layer):
    def __init__(self, n_feat, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.seq=tf.keras.Sequential([
            tf.keras.layers.MaxPooling2D(pool_size=(5,5),strides=1,padding='same'),
            tf.keras.layers.Conv2D(filters=n_feat, kernel_size=3, strides=1, padding='same', use_bias=True)
            ])
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
   
class RefineNet(tf.keras.Model):
    def __init__(self, n_feat, kernel_size, reduction, bias=True, **kwargs):
        super().__init__(name="backbone_refinenet",**kwargs)
        self.stage1=tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=n_feat,kernel_size=kernel_size,strides=1,padding='same',use_bias=bias),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPooling2D()
            ],name='stage1')
        self.cbam1=Residual_CBAM_Block(n_feat, kernel_size, reduction, name='cbam1')
        self.stage2=tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=n_feat*2,kernel_size=kernel_size,strides=1,padding='same',use_bias=bias),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPooling2D()
            ], name='stage2')
        self.cbam2=Residual_CBAM_Block(n_feat*2, kernel_size, reduction//2, name='cbam2')
        self.downsample1=DownSample(n_feat*2, name='downsample1')
        self.downsample2=DownSampleLevel(n_feat*2,2, name='downsample2')
        self.downsample3=DownSampleLevel(n_feat*2,3, name='downsample3')
        self.residual_conv_unit0=tf.keras.Sequential([
            ResidualConvUnit(n_feat*2,kernel_size,bias=True),
            ResidualConvUnit(n_feat*2,kernel_size,bias=True)
            ], name='residual_conv_unit0')
        self.residual_conv_unit1=tf.keras.Sequential([
            ResidualConvUnit(n_feat*2,kernel_size,bias=True),
            ResidualConvUnit(n_feat*2,kernel_size,bias=True)
            ], name='residual_conv_unit1')
        self.residual_conv_unit2=tf.keras.Sequential([
            ResidualConvUnit(n_feat*2,kernel_size,bias=True),
            ResidualConvUnit(n_feat*2,kernel_size,bias=True)
            ], name='residual_conv_unit2')
        self.residual_conv_unit3=tf.keras.Sequential([
            ResidualConvUnit(n_feat*2,kernel_size,bias=True),
            ResidualConvUnit(n_feat*2,kernel_size,bias=True)
            ], name='residual_conv_unit3')
        # self.upsample1=UpSample(n_feat*2, name='upsample1')
        # self.upsample2=UpSampleLevel(n_feat*2,2, name='upsample2')
        # self.upsample3=UpSampleLevel(n_feat*2,3, name='upsample3')
        self.upsample_pixel_shuffler1=UpSamplePixelShuffler(n_feat*2,2, name='upsample_pixel_shuffler1')
        self.upsample_pixel_shuffler2=UpSamplePixelShuffler(n_feat*2,4, name='upsample_pixel_shuffler2')
        self.upsample_pixel_shuffler3=UpSamplePixelShuffler(n_feat*2,8, name='upsample_pixel_shuffler3')
        self.chained_residual_pooling=ChainedResidualPoolingNS(n_feat*2,3, name='chained_residual_pooling')
        self.residual_conv_unit=tf.keras.Sequential([
            ResidualConvUnit(n_feat*2,kernel_size,bias=True),
            ResidualConvUnit(n_feat*2,kernel_size,bias=True)
            ], name='residual_conv_unit')
        self.output_stage1=tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters=n_feat,kernel_size=kernel_size,strides=2,padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU()
            ], name='output_stage1')
        self.output_cbam1=Residual_CBAM_Block(n_feat, kernel_size, reduction, name='output_cbam1')
        self.output_stage2=tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters=n_feat//2,kernel_size=kernel_size,strides=2,padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU()
            ], name='output_stage2')
        self.output_cbam2=Residual_CBAM_Block(n_feat//2, kernel_size, reduction//2, name='output_cbam2')
    def call(self,x,training=None):
        # 256*256*1
        x=self.stage1(x, training=training)
        # 128*128*32
        x=self.cbam1(x, training=training)
        x=self.stage2(x, training=training)
        # 64*64*64
        x=self.cbam2(x, training=training)
        # 64*64*64
        x_1=tf.identity(x)
        # 64*64*64
        # 64*64*64
        x_2=self.downsample1(x, training=training)
        # 32*32*64
        # 64*64*64
        x_3=self.downsample2(x, training=training)
        # 16*16*64
        # 64*64*64
        x_4=self.downsample3(x, training=training)
        # 8*8*64
        x_1=self.residual_conv_unit0(x_1, training=training)
        # 64*64*64
        x_2=self.residual_conv_unit1(x_2, training=training)
        # 32*32*64
        x_3=self.residual_conv_unit2(x_3, training=training)
        # 16*16*64
        x_4=self.residual_conv_unit3(x_4, training=training)
        # 8*8*64
        # x_4=self.upsample3(x_4, training=training)
        x_4=self.upsample_pixel_shuffler3(x_4, training=training)
        # 64*64*64
        # x_3=self.upsample2(x_3, training=training)
        x_3=self.upsample_pixel_shuffler2(x_3, training=training)
        # 64*64*64
        # x_2=self.upsample1(x_2, training=training)
        x_2=self.upsample_pixel_shuffler1(x_2, training=training)
        # 64*64*64
        x_1=tf.identity(x_1)
        # 64*64*64
        x=x_1+x_2+x_3+x_4
        x=self.chained_residual_pooling(x, training=training)
        x=self.residual_conv_unit(x, training=training)
        x=self.output_stage1(x, training=training)
        # 128*128*32
        x=self.output_cbam1(x, training=training)
        # 128*128*32
        x=self.output_stage2(x, training=training)
        # 256*256*16
        x=self.output_cbam2(x, training=training)
        return x
    
class RefineNetWithMoA(tf.keras.models.Model):
    def __init__(self, backbone: tf.keras.models.Model, feat_channels: int, num_experts=4, r_ratio=0.25, k=3, router_hidden=64):
        super().__init__(name="backbone_with_moa")
        self.backbone = backbone
        self.moa      = MoAResidualBlock(feat_channels, num_experts, r_ratio, k, router_hidden, name="moa")
        self.out_head = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters=1,kernel_size=(1,1),strides=(1,1),padding='same'),
            tf.keras.layers.Activation('tanh')
            ], name="out_head")
        self._last_moa_w = None
    def call(self, x, training=None):
        feat = self.backbone(x, training=training)            
        feat_moa, w = self.moa(feat, training=training)       
        self._last_moa_w = w
        y = self.out_head(feat_moa, training=training)      
        return y
    
def charbonnier_loss(y_pred, y_true, eps=1e-3):
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + eps**2))

def psnr_loss(y_pred, y_true, max_val=2.0, eps=1e-8):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    mse = tf.maximum(mse, eps)
    return 10.0 * tf.math.log((max_val ** 2) / mse) / tf.math.log(10.0)

def ssim_loss(y_pred, y_true, max_val=2.0):
    loss_ssim = tf.image.ssim(y_true, y_pred, max_val=max_val)
    loss_ssim=1-tf.reduce_mean(loss_ssim)
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
    """
    归一化熵 Hn = H/ln(E)，逐样本度量均匀度；w:[B,E] 为软权重
    """
    w = tf.clip_by_value(w, eps, 1.0)
    H  = -tf.reduce_sum(w * tf.math.log(w), axis=-1)     # [B]
    E  = tf.cast(tf.shape(w)[-1], w.dtype)
    return H / tf.math.log(E)                            # [B] ∈ [0,1]

def entropy_to_target(w, H_target):
    """
    逐样本防均匀（靠近目标熵）：mean (Hn - H_target)^2
    - 早期: H_target≈两专家（E=4 时 ~0.5），鼓励“别太尖”
    - 后期: H_target→0，鼓励“单专家/稀疏”
    """
    Hn = entropy_norm(w)
    return tf.reduce_mean(tf.square(Hn - H_target))

def kl_to_uniform_batch_mean(w, eps=1e-8):
    """
    跨样本防偏科：KL( mean_b w || Uniform )
    - 把 batch 平均路由分布拉向均匀，避免“全体样本涌向一个专家”
    """
    w_mean = tf.reduce_mean(w, axis=0)                  # [E]
    w_mean = tf.clip_by_value(w_mean, eps, 1.0)
    E = tf.cast(tf.shape(w_mean)[0], w_mean.dtype)
    return tf.reduce_sum(w_mean * (tf.math.log(w_mean) + tf.math.log(E)))

@tf.function
def compute_apply_gradients_backbone_stage0(model,x,y,optimizer):
    with tf.GradientTape() as tape:
        predict = model(x,training=True)
        loss_ssim = ssim_loss(predict,y,max_val=2.0)
        loss_psnr = psnr_loss(predict,y,max_val=2.0)
        loss_ps = loss_ssim/(loss_psnr+1e-8)
        loss_charbonnier = charbonnier_loss(predict,y,eps=1e-8)
        loss=loss_charbonnier+loss_ps*5
        
    gradients=tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    return {"loss":loss,"loss_charb":loss_charbonnier,"loss_ps":loss_ps}

@tf.function
def compute_apply_gradients_moa_stage1(model,x,y,optimizer,lambda_sparse,lambda_diversity,H_target):
    with tf.GradientTape() as tape:
        predict = model(x,training=True)
        loss_ssim = ssim_loss(predict,y,max_val=2.0)
        loss_psnr = psnr_loss(predict,y,max_val=2.0)
        loss_ps = loss_ssim/(loss_psnr+1e-3)
        loss_charbonnier = charbonnier_loss(predict,y,eps=1e-3)
        loss_rec=loss_charbonnier+loss_ps*5
        
        # === 路由正则 ===
        w = getattr(model.moa, "last_w", None)
        if w is not None:
            loss_sparse    = entropy_to_target(w, H_target)
            loss_diversity = kl_to_uniform_batch_mean(w)
            w_mean = tf.reduce_mean(w, axis=0)
            top1 = tf.reduce_mean(tf.reduce_max(w, axis=-1))
            H_bar = tf.reduce_mean(entropy_norm(w))
            KLbm  = kl_to_uniform_batch_mean(w)
        else:
            loss_sparse = tf.constant(0.0, tf.float32)
            loss_diversity = tf.constant(0.0, tf.float32)
            w_mean = tf.zeros([1], tf.float32)
            top1 = tf.constant(0.0, tf.float32)
            H_bar = tf.constant(0.0, tf.float32)
            KLbm = tf.constant(0.0, tf.float32)
        loss=loss_rec+lambda_sparse*loss_sparse+lambda_diversity*loss_diversity
        
    gradients=tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    return {"loss":loss,"loss_rec":loss_rec,"loss_sparse":loss_sparse,"loss_diversity":loss_diversity,
            "w_mean":w_mean,"top1":top1,"w":w,"H_bar":H_bar,"KLbm":KLbm}

@tf.function
def compute_apply_gradients_moa_part_backbone_stage2(model,x,y,optimizer,lambda_sparse,lambda_diversity,H_target):
    with tf.GradientTape() as tape:
        predict = model(x,training=True)
        loss_ssim = ssim_loss(predict,y,max_val=2.0)
        loss_psnr = psnr_loss(predict,y,max_val=2.0)
        loss_ps = loss_ssim/(loss_psnr+1e-3)
        loss_charbonnier = charbonnier_loss(predict,y,eps=1e-3)
        loss_rec=loss_charbonnier+loss_ps*5
        
        # === 路由正则 ===
        w = getattr(model.moa, "last_w", None)
        if w is not None:
            loss_sparse    = entropy_to_target(w, H_target)
            loss_diversity = kl_to_uniform_batch_mean(w)
            w_mean = tf.reduce_mean(w, axis=0)
            top1   = tf.reduce_mean(tf.reduce_max(w, axis=-1))
            H_bar = tf.reduce_mean(entropy_norm(w))
            KLbm  = kl_to_uniform_batch_mean(w)
        else:
            loss_sparse = tf.constant(0.0, tf.float32)
            loss_diversity    = tf.constant(0.0, tf.float32)
            w_mean   = tf.zeros([1], tf.float32)
            top1     = tf.constant(0.0, tf.float32)
            H_bar = tf.constant(0.0, tf.float32)
            KLbm = tf.constant(0.0, tf.float32)

        loss=loss_rec+lambda_sparse*loss_sparse+lambda_diversity*loss_diversity
        
    gradients=tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    return {"loss":loss,"loss_rec":loss_rec,"loss_sparse":loss_sparse,"loss_diversity":loss_diversity,
            "w_mean":w_mean,"top1":top1,"w":w,"H_bar":H_bar,"KLbm":KLbm}

@tf.function
def compute_backbone_stage0(model,x,y):
    predict = model(x,training=False)
    loss_ssim = ssim_loss(predict,y,max_val=2.0)
    loss_psnr = psnr_loss(predict,y,max_val=2.0)
    loss_ps = loss_ssim/(loss_psnr+1e-3)
    loss_charbonnier = charbonnier_loss(predict,y,eps=1e-3)
    loss=loss_charbonnier+loss_ps*5
    return {"loss":loss,"loss_charb":loss_charbonnier,"loss_ps":loss_ps}

@tf.function
def compute_moa_stage1(model,x,y,lambda_sparse,lambda_diversity,H_target):   
    predict = model(x,training=False)
    loss_ssim = ssim_loss(predict,y,max_val=2.0)
    loss_psnr = psnr_loss(predict,y,max_val=2.0)
    loss_ps = loss_ssim/(loss_psnr+1e-3)
    loss_charbonnier = charbonnier_loss(predict,y,eps=1e-3)
    loss_rec=loss_charbonnier+loss_ps*5
        
    # === 路由正则 ===
    w = getattr(model.moa, "last_w", None)
    if w is not None:
        loss_sparse    = entropy_to_target(w, H_target)
        loss_diversity = kl_to_uniform_batch_mean(w)
        w_mean = tf.reduce_mean(w, axis=0)  
        top1   = tf.reduce_mean(tf.reduce_max(w, axis=-1))
        H_bar = tf.reduce_mean(entropy_norm(w))
        KLbm  = kl_to_uniform_batch_mean(w)
    else:
        loss_sparse = tf.constant(0.0, tf.float32)
        loss_diversity    = tf.constant(0.0, tf.float32)
        w_mean   = tf.zeros([1], tf.float32)
        top1     = tf.constant(0.0, tf.float32)
        H_bar     = tf.constant(0.0, tf.float32)
        KLbm     = tf.constant(0.0, tf.float32)

    loss=loss_rec+lambda_sparse*loss_sparse+lambda_diversity*loss_diversity

    return {"loss":loss,"loss_rec":loss_rec,"loss_sparse":loss_sparse,"loss_diversity":loss_diversity,
            "w_mean":w_mean,"top1":top1,"w":w,"H_bar":H_bar,"KLbm":KLbm}

@tf.function
def compute_moa_part_backbone_stage2(model,x,y,lambda_sparse,lambda_diversity,H_target):
    predict = model(x,training=False)
    loss_ssim = ssim_loss(predict,y,max_val=2.0)
    loss_psnr = psnr_loss(predict,y,max_val=2.0)
    loss_ps = loss_ssim/(loss_psnr+1e-3)
    loss_charbonnier = charbonnier_loss(predict,y,eps=1e-3)
    loss_rec=loss_charbonnier+loss_ps*5
        
    # === 路由正则 ===
    w = getattr(model.moa, "last_w", None)
    if w is not None:
        loss_sparse    = entropy_to_target(w, H_target)
        loss_diversity = kl_to_uniform_batch_mean(w)
        w_mean = tf.reduce_mean(w, axis=0)  
        top1   = tf.reduce_mean(tf.reduce_max(w, axis=-1))
        H_bar = tf.reduce_mean(entropy_norm(w))
        KLbm  = kl_to_uniform_batch_mean(w)
    else:
        loss_sparse = tf.constant(0.0, tf.float32)
        loss_diversity    = tf.constant(0.0, tf.float32)
        w_mean   = tf.zeros([1], tf.float32)
        top1     = tf.constant(0.0, tf.float32)
        H_bar     = tf.constant(0.0, tf.float32)
        KLbm     = tf.constant(0.0, tf.float32)

    loss=loss_rec+lambda_sparse*loss_sparse+lambda_diversity*loss_diversity

    return {"loss":loss,"loss_rec":loss_rec,"loss_sparse":loss_sparse,"loss_diversity":loss_diversity,
            "w_mean":w_mean,"top1":top1,"w":w,"H_bar":H_bar,"KLbm":KLbm}

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
    out_path = os.path.join(ROUTER_LOG_DIR, f"router_w_{stage:s}_e{epoch:03d}_b{step_idx:04d}.npy")
    np.save(out_path, w_np.astype(np.float32))
    
def training_mixed(model, training_dataset, validation_dataset, BATCH_SIZE,
                   epochs_stage0, epochs_stage1, epochs_stage2,
                   optimizer_stage0, optimizer_stage1, optimizer_stage2, test_input):
    
    loss_history_stage0=[]
    validation_loss_history_stage0=[]
    loss_history_stage1=[]
    validation_loss_history_stage1=[]
    loss_history_stage2=[]
    validation_loss_history_stage2=[]
    # %%
    # ---------------- STAGE 0: 预训练 backbone + out_head；冻结 MoA ----------------
    freeze_all(model.moa)
    unfreeze(model.backbone)
    model.out_head.trainable = True
    # freeze_bn(model)
    
    loss_epoch_history=[]
    validation_loss_epoch_history=[]
    for epoch in range(epochs_stage0):
        for step_, (x, y) in enumerate(training_dataset):
            metrics = compute_apply_gradients_backbone_stage0(model, x, y, optimizer_stage0)
            loss_epoch_history.append(metrics['loss'].numpy())
            if step_ % 1 == 0:
                print(f"[Stage0][Ep {epoch}] step {step_} loss={to_float(metrics['loss']):.4e}, loss_charb={to_float(metrics['loss_charb']):.4e}, loss_ps={to_float(metrics['loss_ps']):.4e}")
        if epoch%10==0:
            for step_, (vx, vy) in enumerate(validation_dataset):
                val = compute_backbone_stage0(model, vx, vy)
                validation_loss_epoch_history.append(val['loss'].numpy())
                if step_ % 1 == 0:
                    print(f"[Stage0][Val {epoch}]:\n"
                          f"step {step_} loss={to_float(val['loss']):.4e},\n"
                          f"loss_charb={to_float(val['loss_charb']):.4e},\n"
                          f"loss_ps={to_float(val['loss_ps']):.4e}\n")
        if epoch%10==0:
            clutter_inference,clutter_prediction=inference_patch(model,test_input['clutter'],resize_shape=(256,256))
            crts_inference,crts_prediction=inference_patch(model,test_input['crts'],resize_shape=(256,256))
            rebar1_inference,rebar1_prediction=inference_patch(model,test_input['rebar1'],resize_shape=(256,256))
            rebar2_inference,rebar2_prediction=inference_patch(model,test_input['rebar2'],resize_shape=(256,256))
            
            pyplot.figure(1)
            pyplot.subplot(221)
            pyplot.imshow(clutter_inference)
            pyplot.subplot(222)
            pyplot.imshow(crts_inference)
            pyplot.subplot(223)
            pyplot.imshow(rebar1_inference)
            pyplot.subplot(224)
            pyplot.imshow(rebar2_inference)
            pyplot.savefig('./Results/Stage0_Original_%s.png'%epoch)
            
            pyplot.figure(2)
            pyplot.subplot(221)
            pyplot.imshow(clutter_prediction)
            pyplot.subplot(222)
            pyplot.imshow(crts_prediction)
            pyplot.subplot(223)
            pyplot.imshow(rebar1_prediction)
            pyplot.subplot(224)
            pyplot.imshow(rebar2_prediction)
            pyplot.savefig('./Results/Stage0_Prediction_%s.png'%epoch)
            
            model.save('./Weights/Stage0_model_%s'%epoch,save_format='tf')
            model.save_weights(filepath='./Weights/Stage0_model_%s.weights.h5'%epoch)
            
        loss_history_stage0.append(np.mean(loss_epoch_history))
        validation_loss_history_stage0.append(np.mean(validation_loss_epoch_history))
    # %%
    # ---------------- STAGE 1: 冻结 backbone + out_head；只训 MoA ----------------
    freeze_all(model.backbone)
    model.out_head.trainable = False
    unfreeze(model.moa)
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
        
    ema_H   = EMA(alpha=0.9)   # H_bar 的 EMA
    ema_KL  = EMA(alpha=0.9)   # KLbm 的 EMA
    ema_min = EMA(alpha=0.9)   # w_mean 最小负载的 EMA

    entered_phase_c   = False  # 进入 C 段的一次性扰动标记
    warmup_countdown  = 0      # 回暖窗口计数（step）
    WARMUP_STEPS      = 300    # 回暖持续步数
    MIN_LOAD_TH       = 0.10   # 守护阈值（t<0.8 期间启用
    
    loss_epoch_history=[]
    validation_loss_epoch_history=[]
    T = float(epochs_stage1)    # 或 epochs_stage2
    E = tf.cast(getattr(model.moa, "num_experts", 4), tf.float32)
    H_two = tf.math.log(2.0) / tf.math.log(E)   # “两专家”对应的归一化熵
    for epoch in range(epochs_stage1):
        t = epoch / max(1.0, T)     # ∈[0,1)
        if t < 0.4:
            # Phase A: 纯软路由、探索；避免早期塌缩
            model.moa.router.set_topk(0)
            model.moa.router.set_gumbel(True)
            tau = 2.0 - 1.0 * (t / 0.4)             # 2.0 → 1.0
            H_target = H_two                         # 先别太尖（≈两专家）
            lambda_sparse = 1.0
            lambda_div    = 1.0

        elif t < 0.8:
            # Phase B: 软+硬联合；开始分化但不偏科
            model.moa.router.set_topk(2)
            model.moa.router.set_gumbel(True)
            tau = 1.0 - 0.2 * ((t - 0.4) / 0.4)     # 1.0 → 0.8
            # H_target 从“两专家”线性滑向中间（可直接用 0.25~0.3），更稳
            H_target = (1.0 - (t - 0.4) / 0.4) * H_two * 0.6
            lambda_sparse = 1.0
            lambda_div    = 1.0

        else:
            # Phase C: 收敛与部署取向；更稀疏、更专精
            model.moa.router.set_topk(1)
            model.moa.router.set_gumbel(True)
            tau = 0.7 - 0.2 * ((t - 0.8) / 0.2)     # 0.8 → 0.6
            H_target = 0.0                          # 单专家
            lambda_sparse = 1.2                     # 稍加力
            lambda_div    = 1.0

        if (0.6 <= t < 0.8) and (ema_H.value and ema_KL.value):
            if (ema_H.value < 0.60) and (ema_KL.value < 0.03):
                model.moa.router.set_topk(1)
                model.moa.router.set_gumbel(True)
                tau = max(0.55, tau - 0.1)
                H_target = 0.0
                lambda_sparse = max(2.0, float(lambda_sparse))
                lambda_div    = min(0.1, float(lambda_div))
        if (t >= 0.8) and (not entered_phase_c):
            try:
                E = int(getattr(model.moa.router, "fc2").units)  # 专家数
                model.moa.router.fc2.bias.assign_add(
                    1e-3 * tf.random.normal([E])
                    )
            except Exception:
                pass
            entered_phase_c = True
            
        model.moa.router.set_tau(float(tau))
    
        for step_, (x, y) in enumerate(training_dataset):
            metrics = compute_apply_gradients_moa_stage1(model, x, y, optimizer_stage1,
                                                         lambda_sparse=lambda_sparse, 
                                                         lambda_diversity=lambda_div,
                                                         H_target=H_target)
            
            H_bar  = float(metrics.get("H_bar", 0.0))
            KLbm   = float(metrics.get("KLbm", 0.0))
            w_mean = metrics.get("w_mean", None)

            ema_H.update(H_bar)
            ema_KL.update(KLbm)
            if w_mean is not None:
                try:
                    min_load = float(np.min(w_mean))
                except Exception:
                    min_load = float(tf.reduce_min(w_mean))
                ema_min.update(min_load)
            else:
                min_load = 0.0

            # ===== 回暖守护：t<0.8 且最小负载过低时，短暂软化 =====
            if (t < 0.8) and (ema_min.value < MIN_LOAD_TH) and (warmup_countdown == 0):
                warmup_countdown = WARMUP_STEPS

            if warmup_countdown > 0:
                # 完整回暖：取消 top-k、提高温度
                model.moa.router.set_topk(0)
                model.moa.router.set_tau(float(tau + 0.2))
                warmup_countdown -= 1
                
            log_router_weights_from_tensor(metrics.get("w", None), "stage1", epoch, step_)
            loss_epoch_history.append(metrics['loss'].numpy())
            if step_ % 1 == 0:
                print(f"[Stage1][Ep {epoch}]:\n"
                      f"loss={to_float(metrics['loss']):.4e},\n"
                      f"loss_rec={to_float(metrics['loss_rec']):.4e},\n"
                      f"loss_sparse={to_float(metrics['loss_sparse']):.4e},\n"
                      f"loss_diversity={to_float(metrics['loss_diversity']):.4e},\n"
                      f"w_mean={to_list(metrics['w_mean'])},\n"
                      f"top1={to_float(metrics['top1'])},\n"
                      f"tau={float(tau):.2f},\n"
                      f"lam=(sp:{lambda_sparse:+.3f}, dv:{lambda_div:.3f}),\n"
                      f"H_bar={to_float(metrics['H_bar']):.3f},\n"
                      f"KLbm={to_float(metrics['KLbm']):.3f}\n")
            
        if epoch%10==0:
            for step_, (vx, vy) in enumerate(validation_dataset):
                val = compute_moa_stage1(model, vx, vy, 
                                         lambda_sparse=lambda_sparse, 
                                         lambda_diversity=lambda_div,
                                         H_target=H_target)
                validation_loss_epoch_history.append(val['loss'].numpy())
                if step_ % 1 == 0:
                    print(f"[Stage1][Val {epoch}]:\n"
                          f"loss={to_float(metrics['loss']):.4e},\n"
                          f"loss_rec={to_float(metrics['loss_rec']):.4e},\n"
                          f"loss_sparse={to_float(metrics['loss_sparse']):.4e},\n"
                          f"loss_diversity={to_float(metrics['loss_diversity']):.4e},\n"
                          f"w_mean={to_list(metrics['w_mean'])},\n"
                          f"top1={to_float(metrics['top1'])},\n"
                          f"tau={float(tau):.2f},\n"
                          f"lam=(sp:{lambda_sparse:+.3f}, dv:{lambda_div:.3f}),\n"
                          f"H_bar={to_float(metrics['H_bar']):.3f},\n"
                          f"KLbm={to_float(metrics['KLbm']):.3f}\n")
                    
        w_mean = tf.reduce_mean(metrics['w'], axis=0) if 'w' in metrics else None
        if w_mean is not None:
            min_load = float(tf.reduce_min(w_mean))
            # 若出现严重偏科（某专家几乎没人用），临时回暖一轮
            if min_load < 0.05 and t < 0.8:
                print("[Guard] severe imbalance -> temporary soften")
                model.moa.router.set_topk(0)
                model.moa.router.set_tau(min(float(model.moa.router.tau) + 0.2, 2.0))
                
        if epoch%10==0:
            clutter_inference,clutter_prediction=inference_patch(model,test_input['clutter'],resize_shape=(256,256))
            crts_inference,crts_prediction=inference_patch(model,test_input['crts'],resize_shape=(256,256))
            rebar1_inference,rebar1_prediction=inference_patch(model,test_input['rebar1'],resize_shape=(256,256))
            rebar2_inference,rebar2_prediction=inference_patch(model,test_input['rebar2'],resize_shape=(256,256))
            
            pyplot.figure(1)
            pyplot.subplot(221)
            pyplot.imshow(clutter_inference)
            pyplot.subplot(222)
            pyplot.imshow(crts_inference)
            pyplot.subplot(223)
            pyplot.imshow(rebar1_inference)
            pyplot.subplot(224)
            pyplot.imshow(rebar2_inference)
            pyplot.savefig('./Results/Stage1_Original_%s.png'%epoch)
            
            pyplot.figure(2)
            pyplot.subplot(221)
            pyplot.imshow(clutter_prediction)
            pyplot.subplot(222)
            pyplot.imshow(crts_prediction)
            pyplot.subplot(223)
            pyplot.imshow(rebar1_prediction)
            pyplot.subplot(224)
            pyplot.imshow(rebar2_prediction)
            pyplot.savefig('./Results/Stage1_Prediction_%s.png'%epoch)
            
            model.save('./Weights/Stage1_model_%s'%epoch,save_format='tf')
            model.save_weights(filepath='./Weights/Stage1_model_%s.weights.h5'%epoch)
        
        loss_history_stage1.append(np.mean(loss_epoch_history))
        validation_loss_history_stage1.append(np.mean(validation_loss_epoch_history))
    # %%
    # ---------------- STAGE 2: MoA + 顶层主干微调 ----------------
    freeze_all(model.backbone)
    # 只开顶层几块（按你命名的层）
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
    
    ema_H   = EMA(alpha=0.9)   # H_bar 的 EMA
    ema_KL  = EMA(alpha=0.9)   # KLbm 的 EMA
    ema_min = EMA(alpha=0.9)   # w_mean 最小负载的 EMA

    entered_phase_c   = False  # 进入 C 段的一次性扰动标记
    warmup_countdown  = 0      # 回暖窗口计数（step）
    WARMUP_STEPS      = 300    # 回暖持续步数
    MIN_LOAD_TH       = 0.10   # 守护阈值（t<0.8 期间启用
    
    loss_epoch_history=[]
    validation_loss_epoch_history=[]
    T = float(epochs_stage2)    # 或 epochs_stage2
    E = tf.cast(getattr(model.moa, "num_experts", 4), tf.float32)
    H_two = tf.math.log(2.0) / tf.math.log(E)   # “两专家”对应的归一化熵
    for epoch in range(epochs_stage2):
        t = epoch / max(1.0, T)     # ∈[0,1)
        if t < 0.4:
            # Phase A: 纯软路由、探索；避免早期塌缩
            model.moa.router.set_topk(0)
            model.moa.router.set_gumbel(True)
            tau = 2.0 - 1.0 * (t / 0.4)             # 2.0 → 1.0
            H_target = H_two                         # 先别太尖（≈两专家）
            lambda_sparse = 1.0
            lambda_div    = 1.0

        elif t < 0.8:
            # Phase B: 软+硬联合；开始分化但不偏科
            model.moa.router.set_topk(2)
            model.moa.router.set_gumbel(True)
            tau = 1.0 - 0.2 * ((t - 0.4) / 0.4)     # 1.0 → 0.8
            # H_target 从“两专家”线性滑向中间（可直接用 0.25~0.3），更稳
            H_target = (1.0 - (t - 0.4) / 0.4) * H_two * 0.6
            lambda_sparse = 1.0
            lambda_div    = 1.0

        else:
            # Phase C: 收敛与部署取向；更稀疏、更专精
            model.moa.router.set_topk(1)
            model.moa.router.set_gumbel(True)
            tau = 0.7 - 0.2 * ((t - 0.8) / 0.2)     # 0.8 → 0.6
            H_target = 0.0                          # 单专家
            lambda_sparse = 1.2                     # 稍加力
            lambda_div    = 1.0
        
        if (0.6 <= t < 0.8) and (ema_H.value and ema_KL.value):
            if (ema_H.value < 0.60) and (ema_KL.value < 0.03):
                model.moa.router.set_topk(1)
                model.moa.router.set_gumbel(True)
                tau = max(0.55, tau - 0.1)
                H_target = 0.0
                lambda_sparse = max(2.0, float(lambda_sparse))
                lambda_div    = min(0.1, float(lambda_div))
        if (t >= 0.8) and (not entered_phase_c):
            try:
                E = int(getattr(model.moa.router, "fc2").units)  # 专家数
                model.moa.router.fc2.bias.assign_add(
                    1e-3 * tf.random.normal([E])
                    )
            except Exception:
                pass
            entered_phase_c = True
        
        model.moa.router.set_tau(float(tau))
    
        for step_, (x, y) in enumerate(training_dataset):
            metrics = compute_apply_gradients_moa_part_backbone_stage2(model, x, y, optimizer_stage2,
                                                                       lambda_sparse=lambda_sparse, 
                                                                       lambda_diversity=lambda_div,
                                                                       H_target=H_target)
            H_bar  = float(metrics.get("H_bar", 0.0))
            KLbm   = float(metrics.get("KLbm", 0.0))
            w_mean = metrics.get("w_mean", None)

            ema_H.update(H_bar)
            ema_KL.update(KLbm)
            if w_mean is not None:
                try:
                    min_load = float(np.min(w_mean))
                except Exception:
                    min_load = float(tf.reduce_min(w_mean))
                ema_min.update(min_load)
            else:
                min_load = 0.0

            # ===== 回暖守护：t<0.8 且最小负载过低时，短暂软化 =====
            if (t < 0.8) and (ema_min.value < MIN_LOAD_TH) and (warmup_countdown == 0):
                warmup_countdown = WARMUP_STEPS

            if warmup_countdown > 0:
                # 完整回暖：取消 top-k、提高温度
                model.moa.router.set_topk(0)
                model.moa.router.set_tau(float(tau + 0.2))
                warmup_countdown -= 1
        
            log_router_weights_from_tensor(metrics.get("w", None), "stage2", epoch, step_)
            loss_epoch_history.append(metrics['loss'].numpy())
            if step_ % 1 == 0:
                print(f"[Stage2][Ep {epoch}]:\n"
                      f"loss={to_float(metrics['loss']):.4e},\n"
                      f"loss_rec={to_float(metrics['loss_rec']):.4e},\n"
                      f"loss_sparse={to_float(metrics['loss_sparse']):.4e},\n"
                      f"loss_diversity={to_float(metrics['loss_diversity']):.4e},\n"
                      f"w_mean={to_list(metrics['w_mean'])},\n"
                      f"top1={to_float(metrics['top1'])},\n"
                      f"tau={float(tau):.2f},\n"
                      f"lam=(sp:{lambda_sparse:+.3f}, dv:{lambda_div:.3f}),\n"
                      f"H_bar={to_float(metrics['H_bar']):.3f},\n"
                      f"KLbm={to_float(metrics['KLbm']):.3f}\n")
            
        if epoch%10==0:
            for step_, (vx, vy) in enumerate(validation_dataset):
                val = compute_moa_part_backbone_stage2(model, vx, vy, 
                                                       lambda_sparse=lambda_sparse, 
                                                       lambda_diversity=lambda_div,
                                                       H_target=H_target)
                validation_loss_epoch_history.append(val['loss'].numpy())
                if step_ % 1 == 0:
                    print(f"[Stage2][Val {epoch}]:\n"
                          f"loss={to_float(metrics['loss']):.4e},\n"
                          f"loss_rec={to_float(metrics['loss_rec']):.4e},\n"
                          f"loss_sparse={to_float(metrics['loss_sparse']):.4e},\n"
                          f"loss_diversity={to_float(metrics['loss_diversity']):.4e},\n"
                          f"w_mean={to_list(metrics['w_mean'])},\n"
                          f"top1={to_float(metrics['top1'])},\n"
                          f"tau={float(tau):.2f},\n"
                          f"lam=(sp:{lambda_sparse:+.3f}, dv:{lambda_div:.3f}),\n"
                          f"H_bar={to_float(metrics['H_bar']):.3f},\n"
                          f"KLbm={to_float(metrics['KLbm']):.3f}\n")
                    
        w_mean = tf.reduce_mean(metrics['w'], axis=0) if 'w' in metrics else None
        if w_mean is not None:
            min_load = float(tf.reduce_min(w_mean))
            # 若出现严重偏科（某专家几乎没人用），临时回暖一轮
            if min_load < 0.05 and t < 0.8:
                print("[Guard] severe imbalance -> temporary soften")
                model.moa.router.set_topk(0)
                model.moa.router.set_tau(min(float(model.moa.router.tau) + 0.2, 2.0))
                        
        if epoch%10==0:
            clutter_inference,clutter_prediction=inference_patch(model,test_input['clutter'],resize_shape=(256,256))
            crts_inference,crts_prediction=inference_patch(model,test_input['crts'],resize_shape=(256,256))
            rebar1_inference,rebar1_prediction=inference_patch(model,test_input['rebar1'],resize_shape=(256,256))
            rebar2_inference,rebar2_prediction=inference_patch(model,test_input['rebar2'],resize_shape=(256,256))
            
            pyplot.figure(1)
            pyplot.subplot(221)
            pyplot.imshow(clutter_inference)
            pyplot.subplot(222)
            pyplot.imshow(crts_inference)
            pyplot.subplot(223)
            pyplot.imshow(rebar1_inference)
            pyplot.subplot(224)
            pyplot.imshow(rebar2_inference)
            pyplot.savefig('./Results/Stage2_Original_%s.png'%epoch)
            
            pyplot.figure(2)
            pyplot.subplot(221)
            pyplot.imshow(clutter_prediction)
            pyplot.subplot(222)
            pyplot.imshow(crts_prediction)
            pyplot.subplot(223)
            pyplot.imshow(rebar1_prediction)
            pyplot.subplot(224)
            pyplot.imshow(rebar2_prediction)
            pyplot.savefig('./Results/Stage2_Prediction_%s.png'%epoch)
            
            model.save('./Weights/Stage2_model_%s'%epoch,save_format='tf')
            model.save_weights(filepath='./Weights/Stage2_model_%s.weights.h5'%epoch)
            
        loss_history_stage2.append(np.mean(loss_epoch_history))
        validation_loss_history_stage2.append(np.mean(validation_loss_epoch_history))
    return {"model":model,"loss_history_stage0":loss_history_stage0,"loss_history_stage1":loss_history_stage1,
            "loss_history_stage2":loss_history_stage2,"validation_loss_history_stage0":validation_loss_history_stage0,
            "validation_loss_history_stage1":validation_loss_history_stage1,"validation_loss_history_stage2":validation_loss_history_stage2}

def print_layer_tree(layer, depth=0, max_depth=99):
    indent = "  " * depth
    print(f"{indent}{layer.name:40s}  trainable={layer.trainable}")
    if depth >= max_depth: 
        return
    if hasattr(layer, "layers"):
        for sub in layer.layers:
            print_layer_tree(sub, depth+1, max_depth)

if __name__=='__main__':
    ROUTER_LOG_DIR = "./router_w_logs"
    os.makedirs(ROUTER_LOG_DIR, exist_ok=True)
    #%%
    backbone=RefineNet(n_feat=32, kernel_size=3, reduction=8)
    moa_model=RefineNetWithMoA(backbone,feat_channels=16,num_experts=4,r_ratio=0.25,k=3,router_hidden=64)
    #%%
    x=tf.keras.Input(shape=(256,256,1))
    _=moa_model(x)
    moa_model.summary()
    moa_model.backbone.summary()
    #%%
    print_layer_tree(moa_model)             
    print("----- BACKBONE ONLY -----")
    print_layer_tree(moa_model.backbone)       
    #%%
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
    train_input=np.load('data_tanh_input.npy').astype('float32')
    train_output=np.load('data_tanh_output.npy').astype('float32')
    validation_input=train_input[::10]
    validation_output=train_output[::10]
    
    TOTAL_SAMPLES = 6567
    BATCH_SIZE = 16
    TRAIN_BUF=TOTAL_SAMPLES*2
    training_dataset=tf.data.Dataset.from_tensor_slices((train_input,train_output)).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
    validation_dataset=tf.data.Dataset.from_tensor_slices((validation_input,validation_output)).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
    # %%
    # BATCH_SIZE=32
    epochs_stage0=101
    epochs_stage1=51
    epochs_stage2=51
    steps_per_epoch=int(tf.math.ceil(TOTAL_SAMPLES/BATCH_SIZE))
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=2e-4,
    decay_steps=steps_per_epoch*5,
    decay_rate=0.9)
    optimizer_stage0 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # optimizer_stage0 = tf.keras.optimizers.Adam(learning_rate=2e-4,clipnorm=1.0)
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=steps_per_epoch*5,
    decay_rate=0.9)
    optimizer_stage1 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # optimizer_stage1 = tf.keras.optimizers.Adam(learning_rate=2e-4,clipnorm=1.0)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=steps_per_epoch*5,
    decay_rate=0.9)
    optimizer_stage2 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # optimizer_stage2 = tf.keras.optimizers.Adam(learning_rate=1e-4,clipnorm=1.0)
    
    test_input_clutter=np.load('00 clutter_testing.npy').astype('float32')
    test_input_crts=np.load('01 crts_testing.npy').astype('float32')
    test_input_rebar1=np.load('02 rebar_testing.npy').astype('float32')
    test_input_rebar2=np.load('03 rebar_testing.npy').astype('float32')
    test_input={'clutter':test_input_clutter,'crts':test_input_crts,'rebar1':test_input_rebar1,'rebar2':test_input_rebar2}
    model_info=training_mixed(moa_model, training_dataset, validation_dataset, BATCH_SIZE,
                              epochs_stage0, epochs_stage1, epochs_stage2,
                              optimizer_stage0, optimizer_stage1, optimizer_stage2, test_input)