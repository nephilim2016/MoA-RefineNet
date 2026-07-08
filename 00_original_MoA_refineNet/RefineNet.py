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
def compute_apply_gradients_moa_stage1(model,x,y,optimizer,lambda_sparse,lambda_diversity):
    with tf.GradientTape() as tape:
        predict = model(x,training=True)
        loss_ssim = ssim_loss(predict,y,max_val=2.0)
        loss_psnr = psnr_loss(predict,y,max_val=2.0)
        loss_ps = loss_ssim/(loss_psnr+1e-3)
        loss_charbonnier = charbonnier_loss(predict,y,eps=1e-3)
        loss_rec=loss_charbonnier+loss_ps*5
        
        # w=getattr(model,"_last_moa_w", None)
        # loss_sparse = entropy_loss(w) if w is not None else 0.0
        # loss_diversity = diversity_loss(w) if w is not None else 0.0
        
        # === 路由正则 ===
        w = getattr(model.moa, "last_w", None)
        if w is not None:
            w_clip = tf.clip_by_value(w, 1e-8, 1.0)
            loss_sparse = entropy_loss(w_clip)
            loss_diversity = diversity_loss(w)
            w_mean = tf.reduce_mean(w, axis=0)
            top1 = tf.reduce_mean(tf.reduce_max(w, axis=-1))
        else:
            loss_sparse = tf.constant(0.0, tf.float32)
            loss_diversity = tf.constant(0.0, tf.float32)
            w_mean = tf.zeros([1], tf.float32)
            top1 = tf.constant(0.0, tf.float32)
        loss=loss_rec+lambda_sparse*loss_sparse+lambda_diversity*loss_diversity
        
    gradients=tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    return {"loss":loss,"loss_rec":loss_rec,"loss_sparse":loss_sparse,"loss_diversity":loss_diversity,"w_mean":w_mean,"top1":top1,"w":w}

@tf.function
def compute_apply_gradients_moa_part_backbone_stage2(model,x,y,optimizer,lambda_sparse,lambda_diversity):
    with tf.GradientTape() as tape:
        predict = model(x,training=True)
        loss_ssim = ssim_loss(predict,y,max_val=2.0)
        loss_psnr = psnr_loss(predict,y,max_val=2.0)
        loss_ps = loss_ssim/(loss_psnr+1e-3)
        loss_charbonnier = charbonnier_loss(predict,y,eps=1e-3)
        loss_rec=loss_charbonnier+loss_ps*5
        
        # w=getattr(model,"_last_moa_w", None)
        # loss_sparse = entropy_loss(w) if w is not None else 0.0
        # loss_diversity = diversity_loss(w) if w is not None else 0.0
        
        # === 路由正则 ===
        w = getattr(model.moa, "last_w", None)
        if w is not None:
            w_clip = tf.clip_by_value(w, 1e-8, 1.0)
            loss_sparse = entropy_loss(w_clip)
            loss_diversity  = diversity_loss(w)
            w_mean = tf.reduce_mean(w, axis=0)
            top1   = tf.reduce_mean(tf.reduce_max(w, axis=-1))
        else:
            loss_sparse = tf.constant(0.0, tf.float32)
            loss_diversity    = tf.constant(0.0, tf.float32)
            w_mean   = tf.zeros([1], tf.float32)
            top1     = tf.constant(0.0, tf.float32)

        loss=loss_rec+lambda_sparse*loss_sparse+lambda_diversity*loss_diversity
        
    gradients=tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    return {"loss":loss,"loss_rec":loss_rec,"loss_sparse":loss_sparse,"loss_diversity":loss_diversity,"w_mean":w_mean,"top1":top1,"w":w}

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
def compute_moa_stage1(model,x,y,lambda_sparse,lambda_diversity):   
    predict = model(x,training=False)
    loss_ssim = ssim_loss(predict,y,max_val=2.0)
    loss_psnr = psnr_loss(predict,y,max_val=2.0)
    loss_ps = loss_ssim/(loss_psnr+1e-3)
    loss_charbonnier = charbonnier_loss(predict,y,eps=1e-3)
    loss_rec=loss_charbonnier+loss_ps*5
        
    # w=getattr(model,"_last_moa_w", None)
    # loss_sparse = entropy_loss(w) if w is not None else 0.0
    # loss_diversity = diversity_loss(w) if w is not None else 0.0
        
    # === 路由正则 ===
    w = getattr(model.moa, "last_w", None)
    if w is not None:
        w_clip = tf.clip_by_value(w, 1e-8, 1.0)
        loss_sparse = entropy_loss(w_clip)
        loss_diversity  = diversity_loss(w)
        w_mean = tf.reduce_mean(w, axis=0)  
        top1   = tf.reduce_mean(tf.reduce_max(w, axis=-1))
    else:
        loss_sparse = tf.constant(0.0, tf.float32)
        loss_diversity    = tf.constant(0.0, tf.float32)
        w_mean   = tf.zeros([1], tf.float32)
        top1     = tf.constant(0.0, tf.float32)

    loss=loss_rec+lambda_sparse*loss_sparse+lambda_diversity*loss_diversity

    return {"loss":loss,"loss_rec":loss_rec,"loss_sparse":loss_sparse,"loss_diversity":loss_diversity,"w_mean":w_mean,"top1":top1}

@tf.function
def compute_moa_part_backbone_stage2(model,x,y,lambda_sparse,lambda_diversity):
    predict = model(x,training=False)
    loss_ssim = ssim_loss(predict,y,max_val=2.0)
    loss_psnr = psnr_loss(predict,y,max_val=2.0)
    loss_ps = loss_ssim/(loss_psnr+1e-3)
    loss_charbonnier = charbonnier_loss(predict,y,eps=1e-3)
    loss_rec=loss_charbonnier+loss_ps*5
        
    # w=getattr(model,"_last_moa_w", None)
    # loss_sparse = entropy_loss(w) if w is not None else 0.0
    # loss_diversity = diversity_loss(w) if w is not None else 0.0
        
    # === 路由正则 ===
    w = getattr(model.moa, "last_w", None)
    if w is not None:
        w_clip = tf.clip_by_value(w, 1e-8, 1.0)
        loss_sparse = entropy_loss(w_clip)
        loss_diversity  = diversity_loss(w)
        w_mean = tf.reduce_mean(w, axis=0)  
        top1   = tf.reduce_mean(tf.reduce_max(w, axis=-1))
    else:
        loss_sparse = tf.constant(0.0, tf.float32)
        loss_diversity    = tf.constant(0.0, tf.float32)
        w_mean   = tf.zeros([1], tf.float32)
        top1     = tf.constant(0.0, tf.float32)

    loss=loss_rec+lambda_sparse*loss_sparse+lambda_diversity*loss_diversity

    return {"loss":loss,"loss_rec":loss_rec,"loss_sparse":loss_sparse,"loss_diversity":loss_diversity,"w_mean":w_mean,"top1":top1}

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
    ###############################################################################
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
                    print(f"[Stage0][Val {epoch}] step {step_} loss={to_float(val['loss']):.4e}, loss_charb={to_float(val['loss_charb']):.4e}, loss_ps={to_float(val['loss_ps']):.4e}")
        if epoch%10==0:
            clutter_inference,clutter_prediction=inference_patch(model,test_input['clutter'],resize_shape=(256,512))
            crts_inference,crts_prediction=inference_patch(model,test_input['crts'],resize_shape=(256,512))
            rebar1_inference,rebar1_prediction=inference_patch(model,test_input['rebar1'],resize_shape=(256,512))
            rebar2_inference,rebar2_prediction=inference_patch(model,test_input['rebar2'],resize_shape=(256,512))
            
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
    ##############################################################################
    # ---------------- STAGE 1: 冻结 backbone + out_head；只训 MoA ----------------
    freeze_all(model.backbone)
    model.out_head.trainable = False
    unfreeze(model.moa)
    # freeze_bn(model)  
    
    loss_epoch_history=[]
    validation_loss_epoch_history=[]
    E = getattr(model.moa, "num_experts", 5)
    MAX_ENT = tf.math.log(tf.cast(E, tf.float32))
    EPOCHS_WARMUP = max(5, epochs_stage1 // 10)  # 例如前 1/3 用 warm-up
    for epoch in range(epochs_stage1):
        # ---- 基础调度（先稳后尖） ----
        if epoch < EPOCHS_WARMUP:
            tau = 1.5                                 # 稍高温
            lambda_sparse = 0.0                       # 不推稀疏
            lambda_div    = 0.0
            model.moa.router.set_topk(0)          # 暂不启 top-k
            model.moa.router.set_gumbel(False)
        else:
            # 线性把 tau 拉到 0.8，再逐步到 0.6
            t = (epoch - EPOCHS_WARMUP) / max(1, epochs_stage1 - EPOCHS_WARMUP)
            tau = 0.8 - 0.2 * float(tf.clip_by_value(t, 0.0, 1.0))   # 0.8 → 0.6
            # 稀疏权 0 → 0.3 线性升温（你也可以设到 0.5）
            lambda_sparse = 0.30 * float(tf.clip_by_value(t, 0.0, 1.0))
            lambda_div    = 0.15                                     # 保持总体均衡
            model.moa.router.set_topk(2)                         # 关键：top-k
            model.moa.router.set_gumbel(True)                    # 带点探索

        model.moa.router.set_tau(float(tau))
    
        for step_, (x, y) in enumerate(training_dataset):
            metrics = compute_apply_gradients_moa_stage1(model, x, y, optimizer_stage1,
                                                         lambda_sparse=lambda_sparse, 
                                                         lambda_diversity=lambda_div)
            log_router_weights_from_tensor(metrics.get("w", None), "stage1", epoch, step_)
            loss_epoch_history.append(metrics['loss'].numpy())
            if step_ % 1 == 0:
                print(f"[Stage1][Ep {epoch}] loss={to_float(metrics['loss']):.4e}, loss_rec={to_float(metrics['loss_rec']):.4e}, loss_sparse={to_float(metrics['loss_sparse']):.4e}, loss_diversity={to_float(metrics['loss_diversity']):.4e}, w_mean={to_list(metrics['w_mean'])}, top1={to_float(metrics['top1'])}, tau={float(tau):.2f}, lam=(sp:{lambda_sparse:+.3f}, dv:{lambda_div:.3f})")
        H = float(metrics['loss_sparse'].numpy())      # 就是平均熵
        top1 = float(metrics['top1'].numpy())
        if (H > 0.92 * float(MAX_ENT)) or (top1 < 0.35 and epoch >= EPOCHS_WARMUP):
            print("[Guard] router too uniform -> sharpen more")
            model.moa.router.set_tau(0.6)          # 直接降温
            model.moa.router.set_topk(2)           # 保持 top-k
            lambda_sparse = max(lambda_sparse, 0.4)    # 稍加把劲
            lambda_div    = 0.15                        # 维持均衡不塌到单专家
            
        if epoch%10==0:
            for step_, (vx, vy) in enumerate(validation_dataset):
                val = compute_moa_stage1(model, vx, vy, lambda_sparse=lambda_sparse, lambda_diversity=lambda_div)
                validation_loss_epoch_history.append(val['loss'].numpy())
                if step_ % 1 == 0:
                    print(f"[Stage1][Val {epoch}] loss={to_float(val['loss']):.4e}, loss_rec={to_float(val['loss_rec']):.4e}, loss_sparse={to_float(val['loss_sparse']):.4e}, loss_diversity={to_float(val['loss_diversity']):.4e}, w_mean={to_list(metrics['w_mean'])}, top1={to_float(metrics['top1'])}, tau={float(tau):.2f}, lam=(sp:{lambda_sparse:+.3f}, dv:{lambda_div:.3f})")
        if epoch%10==0:
            clutter_inference,clutter_prediction=inference_patch(model,test_input['clutter'],resize_shape=(256,512))
            crts_inference,crts_prediction=inference_patch(model,test_input['crts'],resize_shape=(256,512))
            rebar1_inference,rebar1_prediction=inference_patch(model,test_input['rebar1'],resize_shape=(256,512))
            rebar2_inference,rebar2_prediction=inference_patch(model,test_input['rebar2'],resize_shape=(256,512))
            
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
    ##############################################################
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

    loss_epoch_history=[]
    validation_loss_epoch_history=[]
    E = getattr(model.moa, "num_experts", 5)
    MAX_ENT = tf.math.log(tf.cast(E, tf.float32))
    EPOCHS_WARMUP = max(5, epochs_stage1 // 10)  # 例如前 1/3 用 warm-up
    for epoch in range(epochs_stage1):
        # ---- 基础调度（先稳后尖） ----
        if epoch < EPOCHS_WARMUP:
            tau = 1.5                                 # 稍高温
            lambda_sparse = 0.0                       # 不推稀疏
            lambda_div    = 0.0
            model.moa.router.set_topk(0)          # 暂不启 top-k
            model.moa.router.set_gumbel(False)
        else:
            # 线性把 tau 拉到 0.8，再逐步到 0.6
            t = (epoch - EPOCHS_WARMUP) / max(1, epochs_stage1 - EPOCHS_WARMUP)
            tau = 0.8 - 0.2 * float(tf.clip_by_value(t, 0.0, 1.0))   # 0.8 → 0.6
            # 稀疏权 0 → 0.3 线性升温（你也可以设到 0.5）
            lambda_sparse = 0.30 * float(tf.clip_by_value(t, 0.0, 1.0))
            lambda_div    = 0.15                                     # 保持总体均衡
            model.moa.router.set_topk(2)                         # 关键：top-k
            model.moa.router.set_gumbel(True)                    # 带点探索

        model.moa.router.set_tau(float(tau))
    
        for step_, (x, y) in enumerate(training_dataset):
            metrics = compute_apply_gradients_moa_part_backbone_stage2(model, x, y, optimizer_stage2,
                                                                       lambda_sparse=lambda_sparse, 
                                                                       lambda_diversity=lambda_div)
            log_router_weights_from_tensor(metrics.get("w", None), "stage2", epoch, step_)
            loss_epoch_history.append(metrics['loss'].numpy())
            if step_ % 1 == 0:
                print(f"[Stage2][Ep {epoch}] loss={to_float(metrics['loss']):.4e}, loss_rec={to_float(metrics['loss_rec']):.4e}, loss_sparse={to_float(metrics['loss_sparse']):.4e}, loss_diversity={to_float(metrics['loss_diversity']):.4e}, w_mean={to_list(metrics['w_mean'])}, top1={to_float(metrics['top1'])}, tau={float(tau):.2f}, lam=(sp:{lambda_sparse:+.3f}, dv:{lambda_div:.3f})")
        # —— 简单“去塌回退”：若检测到独占，临时回暖几轮 —— 
        # 条件：top1>0.95 且有专家 w_mean < 0.02
        H = float(metrics['loss_sparse'].numpy())      # 就是平均熵
        top1 = float(metrics['top1'].numpy())
        if (H > 0.92 * float(MAX_ENT)) or (top1 < 0.35 and epoch >= EPOCHS_WARMUP):
            print("[Guard] router too uniform -> sharpen more")
            model.moa.router.set_tau(0.6)          # 直接降温
            model.moa.router.set_topk(2)           # 保持 top-k
            lambda_sparse = max(lambda_sparse, 0.4)    # 稍加把劲
            lambda_div    = 0.15                        # 维持均衡不塌到单专家
        if epoch%10==0:
            for step_, (vx, vy) in enumerate(validation_dataset):
                val = compute_moa_part_backbone_stage2(model, vx, vy, lambda_sparse=lambda_sparse, lambda_diversity=lambda_div)
                validation_loss_epoch_history.append(val['loss'].numpy())
                if step_ % 1 == 0:
                    print(f"[Stage2][Val {epoch}] loss={to_float(val['loss']):.4e}, loss_rec={to_float(val['loss_rec']):.4e}, loss_sparse={to_float(val['loss_sparse']):.4e}, loss_diversity={to_float(val['loss_diversity']):.4e}, w_mean={to_list(metrics['w_mean'])}, top1={to_float(metrics['top1'])}, tau={float(tau):.2f}, lam=(sp:{lambda_sparse:+.3f}, dv:{lambda_div:.3f}")
        
        if epoch%10==0:
            clutter_inference,clutter_prediction=inference_patch(model,test_input['clutter'],resize_shape=(256,512))
            crts_inference,crts_prediction=inference_patch(model,test_input['crts'],resize_shape=(256,512))
            rebar1_inference,rebar1_prediction=inference_patch(model,test_input['rebar1'],resize_shape=(256,512))
            rebar2_inference,rebar2_prediction=inference_patch(model,test_input['rebar2'],resize_shape=(256,512))
            
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