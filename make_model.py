import logging
import os
import random
import numpy as np
# from threading import local
from model.backbones.vit_pytorch import deit_tiny_patch16_224_TransReID, part_attention_deit_small, part_attention_deit_tiny, part_attention_vit_base, part_attention_vit_base_p32, part_attention_vit_large, part_attention_vit_small, vit_base_patch32_224_TransReID, vit_large_patch16_224_TransReID
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones.transformer import TransformerDecoder, TransformerDecoderLayer
from .backbones.resnet import BasicBlock, ResNet, Bottleneck
from random import randint
from PIL import Image
import PIL
import torchvision

import matplotlib.pyplot as plt

from .backbones import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
# from train import nb_domain
# from train import nb_domain
from torch_kmeans import KMeans
from torch_kmeans.utils.distances import (
    BaseDistance,
    CosineSimilarity,
    DotProductSimilarity,
    LpDistance,
)
# alter this to your pre-trained file name
lup_path_name = {
    'vit_base_patch16_224_TransReID': 'vit_base_ics_cfs_lup.pth',
    'vit_small_patch16_224_TransReID': 'vit_base_ics_cfs_lup.pth',
    'deit_nase': 'deit_base_distilled_patch16_224-df68dfff.pth'
}

# alter this to your pre-trained file name
imagenet_path_name = {
    'vit_large_patch16_224_TransReID': 'jx_vit_large_p16_224-4ee7a4dc.pth',
    'vit_base_patch16_224_TransReID': 'jx_vit_base_p16_224-80ecf9dd.pth',
    'vit_base_patch32_224_TransReID': 'jx_vit_base_patch32_224_in21k-8db57226.pth',
    'deit_base_patch16_224_TransReID': 'deit_base_distilled_patch16_224-df68dfff.pth',
    'vit_small_patch16_224_TransReID': 'vit_small_p16_224-15ec54c9.pth',
    'deit_small_patch16_224_TransReID': 'deit_small_distilled_patch16_224-649709d9.pth',
    'deit_ase': 'deit_base_distilled_patch16_224-df68dfff.pth',
    
    'deit_tiny_patch16_224_TransReID': 'deit_tiny_distilled_patch16_224-b40b3cf7.pth'
}


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):

    def __init__(self, model_name, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path_base = cfg.MODEL.PRETRAIN_PATH
        
        # model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 2048
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[2, 2, 2, 2])
            model_path = os.path.join(model_path_base, \
                "resnet18-f37072fd.pth")
            print('using resnet18 as a backbone')
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
            model_path = os.path.join(model_path_base, \
                "resnet34-b627a593.pth")
            print('using resnet34 as a backbone')
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            model_path = os.path.join(model_path_base, \
                "resnet50-0676ba61.pth")
            print('using resnet50 as a backbone')
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 23, 3])
            model_path = os.path.join(model_path_base, \
                "resnet101-63fe2227.pth")
            print('using resnet101 as a backbone')
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            model_path = os.path.join(model_path_base, \
                "resnet152-394f9c45.pth")
            print('using resnet152 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        # self.pool = nn.Linear(in_features=16*8, out_features=1, bias=False)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)  # B, C, h, w
        
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        # global_feat = self.pool(x.flatten(2)).squeeze() # is GAP harming generalization?

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            if 'classifier' in i:  # drop classifier
                continue
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('PAT.train')
        logger.info("Number of parameter: %.2fM" % (total / 1e6))


class build_vit(nn.Module):

    def __init__(self, num_classes, cfg, factory):
        super(build_vit, self).__init__()
        self.cfg = cfg
        model_path_base = cfg.MODEL.PRETRAIN_PATH
        path = imagenet_path_name[cfg.MODEL.TRANSFORMER_TYPE]
        self.model_path = os.path.join(model_path_base, path)
        self.pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: vit as a backbone')

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE]\
            (img_size=cfg.INPUT.SIZE_TRAIN,
            stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate=cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        elif cfg.MODEL.TRANSFORMER_TYPE == 'deit_tiny_patch16_224_TransReID':
            self.in_planes = 192
        elif cfg.MODEL.TRANSFORMER_TYPE == 'vit_large_patch16_224_TransReID':
            self.in_planes = 1024
        if self.pretrain_choice == 'imagenet':
            self.base.load_param(self.model_path)
            print('Loading pretrained ImageNet model......from {}'.format(self.model_path))
            
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.base(x)  # B, N, C
        global_feat = x[:, 0]  # cls token for global feature

        feat = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            return feat if self.neck_feat == 'after' else global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:  # drop classifier
                continue
            if 'bottleneck' in i:
                continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading trained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('PAT.train')
        logger.info("Number of parameter: %.2fM" % (total / 1e6))

'''
part attention vit
'''


class build_part_attention_vit(nn.Module):

    def __init__(self, num_classes, cfg, factory, pretrain_tag='imagenet'):
        super().__init__()
        self.cfg = cfg
        model_path_base = cfg.MODEL.PRETRAIN_PATH
        if pretrain_tag == 'lup':
            path = lup_path_name[cfg.MODEL.TRANSFORMER_TYPE]
        else:
            path = imagenet_path_name[cfg.MODEL.TRANSFORMER_TYPE]
        self.model_path = os.path.join(model_path_base, path)
        self.pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: part token vit as a backbone')

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE]\
            (img_size=cfg.INPUT.SIZE_TRAIN,
            stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate=cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
            pretrain_tag=pretrain_tag)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        elif cfg.MODEL.TRANSFORMER_TYPE == 'deit_tiny_patch16_224_TransReID':
            self.in_planes = 192
        elif cfg.MODEL.TRANSFORMER_TYPE == 'vit_large_patch16_224_TransReID':
            self.in_planes = 1024
        if self.pretrain_choice == 'imagenet':
            self.base.load_param(self.model_path)
            print('Loading pretrained ImageNet model......from {}'.format(self.model_path))

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        layerwise_tokens = self.base(x)  # B, N, C
        layerwise_cls_tokens = [t[:, 0] for t in layerwise_tokens]  # cls token
        part_feat_list = layerwise_tokens[-1][:, 1: 4]  # 3, 768

        layerwise_part_tokens = [[t[:, i] for i in range(1, 4)] for t in layerwise_tokens]  # 12 3 768
        feat = self.bottleneck(layerwise_cls_tokens[-1])

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, layerwise_cls_tokens, layerwise_part_tokens
        else:
            return feat if self.neck_feat == 'after' else layerwise_cls_tokens[-1]

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:  # drop classifier
                continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading trained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('PAT.train')
        logger.info("Number of parameter: %.2fM" % (total / 1e6))        
####CTC reid##333333333333


class build_ctc_vit(nn.Module):

    def __init__(self, num_classes, cfg, factory, pretrain_tag='imagenet', nb_domain=None):
        super().__init__()
        self.cfg = cfg
        model_path_base = cfg.MODEL.PRETRAIN_PATH
        if pretrain_tag == 'lup':
            path = lup_path_name[cfg.MODEL.TRANSFORMER_TYPE]
        else:
            path = imagenet_path_name[cfg.MODEL.TRANSFORMER_TYPE]
        self.model_path = os.path.join(model_path_base, path)
        self.pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.nb_sem = cfg.MODEL.NB_SEM
        print('using Transformer_type: part token vit as a backbone')

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.input_shape = cfg.INPUT.SIZE_TRAIN
        self.input_H = self.input_shape[0]
        self.input_W = self.input_shape[1]
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE]\
            (img_size=cfg.INPUT.SIZE_TRAIN,
            stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate=cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
            pretrain_tag=pretrain_tag,
            cfg=cfg)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        elif cfg.MODEL.TRANSFORMER_TYPE == 'deit_tiny_patch16_224_TransReID':
            self.in_planes = 192
        elif cfg.MODEL.TRANSFORMER_TYPE == 'vit_large_patch16_224_TransReID':
            self.in_planes = 1024
        if self.pretrain_choice == 'imagenet':
            self.base.load_param(self.model_path)
            print('Loading pretrained ImageNet model......from {}'.format(self.model_path))
        self.nb_query = self.nb_sem + 2  # # classification(0)  and part_cls query(1:nb_sem-1) and seg query (nb_sem-1)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.part_classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.grl = GradientReversalLayer()
        self.D_classifier = nn.Linear(self.in_planes, nb_domain, bias=False)
        # self.DecoderLayer = TransformerDecoderLayer(d_model=self.in_planes, nhead=8)
        # self.Decoder = TransformerDecoder(self.DecoderLayer, num_layers=6)
        self.part_classify_token = nn.Parameter(torch.zeros(1, 1, self.in_planes))  # not used
        self.part_cls_query = nn.Parameter(torch.zeros(1, self.nb_sem, self.in_planes))
        self.cls_query = nn.Parameter(torch.zeros(1, 1, self.in_planes))
        self.seg_query = nn.Parameter(torch.zeros(1, 1, self.in_planes))
        
        # self.seg_transform = nn.Sequential(nn.Linear(self.in_planes, int(cfg.INPUT.SIZE_TRAIN[0] / 2 * \
        #                                                                                             cfg.INPUT.SIZE_TRAIN[1] / 2)),
                                                                    
        #                            )
        self.part_classify_token = nn.Parameter(torch.zeros(1, 1, self.in_planes))  # not used
        # self.upsample = nn.Sequential(deconv2d_bn(int(self.in_planes),256),
        #                             deconv2d_bn(256,256),
        #                             deconv2d_bn(256,256),
        #                             deconv2d_bn(256,256))
        self.prob_depth=self.nb_sem+1
        self.dd=nn.Sequential(
                                                                nn.Conv2d(self.in_planes,int(256),1,1),
                                                                nn.LeakyReLU(0.1),
                                                                nn.BatchNorm2d(256 ),
                                                                
                                                                
                                                 
                                                                
                                             
                                             
                                   )
        self.cls_seg=nn.Sequential(
                                                    nn.Conv2d(self.in_planes,int(256),1,1),
                                                                nn.LeakyReLU(0.1),
                                                                nn.BatchNorm2d(256 ),
                                                                
                                                                nn.Conv2d(256,int(256),1,1),
                                                                nn.LeakyReLU(0.1),
                                                                nn.BatchNorm2d(256 ),
                                                                
                                                                nn.Conv2d(256 ,int(self.prob_depth),1,1),
                                                 
                                                                
                                             
                                             
                                   )
        self.random_imgclass_list=os.listdir(os.path.join('data/datasets/Linnaeus/train'))
        self.save_t=0
        self.kmean=KMeans(distance=CosineSimilarity,n_clusters=self.nb_sem+1,verbose=False)
        self.p_transform=[torchvision.transforms.ColorJitter(0.5,0,0,0),
                                torchvision.transforms.ColorJitter(0.,0.5,0,0),
                            torchvision.transforms.ColorJitter(0,0,0.5,0),
                            torchvision.transforms.ColorJitter(0,0,0,0.5),
                          torchvision.transforms.GaussianBlur(5),
                          torchvision.transforms.Grayscale(3)]
        self.g_transform=[torchvision.transforms.transforms.F.crop,
                          torchvision.transforms.transforms.F.hflip]
        self.resize=torchvision.transforms.Resize((cfg.INPUT.SIZE_TRAIN[0],cfg.INPUT.SIZE_TRAIN[1]))
        self.resize_mem=torchvision.transforms.Resize((cfg.INPUT.SIZE_TRAIN[0]//16,cfg.INPUT.SIZE_TRAIN[1]//16))
        
    def forward(self, x,seg_train=False):
        b, c, h, w = x.shape
        mem_H = int(self.input_H / self.cfg.MODEL.STRIDE_SIZE[0])
        mem_W = int(self.input_W / self.cfg.MODEL.STRIDE_SIZE[1])
        feature_map_shape = (b, mem_H, mem_W, self.in_planes)
        if seg_train:
            p_transform1=random.choice(self.p_transform)
            p_transform2=random.choice(self.p_transform)
            g_transform=random.choice(self.g_transform)
            if 'crop' in str(g_transform):
                crop_arg=(random.randint(0,h//2),random.randint(0,w//2),random.randint(h//4,h//2),random.randint(h//4,w//2))
            else:
                pass                
            p1=p_transform1(x)
            p2=p_transform2(x)
            p1 = self.base(p1)[:,1:,:].reshape(feature_map_shape).permute(0, 3, 1, 2)
            if 'crop' in str(g_transform):
                p2 = g_transform(p2,*crop_arg)
                p2=self.resize(p2)
                p1 = g_transform(p1,*crop_arg)
                p1=self.resize_mem(p1)
                
                
            else:
                p2 = g_transform(p2)
                p1 = g_transform(p1)
                
            p2 = self.base(p2)[:,1:,:].reshape(feature_map_shape).permute(0, 3, 1, 2)
            
            p1_logit=self.cls_seg(p1).permute(0, 2,3,1)
            p2_logit=self.cls_seg(p2).permute(0, 2,3,1)
            # p1_kmean=KMeans(distance=CosineSimilarity,n_clusters=self.nb_sem+1,verbose=False)
            # p2_kmean=KMeans(distance=CosineSimilarity,n_clusters=self.nb_sem+1,verbose=False)
            p1_result=self.kmean(p1.reshape(b,-1,self.in_planes).detach())
            p2_result=self.kmean(p2.reshape(b,-1,self.in_planes).detach())
            p1_plabel=p1_result.labels
            p2_plabel=p2_result.labels
            p1_centroid=p1_result.centers
            p2_centroid=p1_result.centers
            if randint(0,300)==1:
                color_map=np.array([[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255]])
                image = np.zeros((256, 128, 3), dtype=np.uint8)
                seg_mask=torch.argmax(p1_logit, dim=3).detach()
                for i in range(256):
                    for j in range(128):
                        color_index = seg_mask[0][int(i/16),int(j/16)] # Subtract 1 because indices start from 0
                        image[i, j] = color_map[color_index]
                input=x[0,::].permute(1,2,0).cpu().detach().numpy()
                input= (input-np.min(input))/(np.max(input)-np.min(input))*255
                image=image*0.2+input*0.8
                # plt.imshow(image)
                image = np.uint8(image)
                image = Image.fromarray(image)
                image.save(os.path.join('exp_2',f"saved{self.save_t}.png"))
                self.save_t+=1
                print('saved img')
            return {
                'p1_map':p1,
                'p2_map':p2,
                'p1_logit':p1_logit,
                    'p2_logit':p2_logit,
                    'p1_plabel':p1_plabel,
                    'p2_plabel':p2_plabel,
                    'p1_centroid':p1_centroid,
                    'p2_centroid':p2_centroid
                    }
        layerwise_tokens = self.base(x)  # B, N, C   64,132,768
        # layerwise_cls_tokens =layerwise_tokens[:, 0] # cls token
        encoder_out = layerwise_tokens[0]  # without classification output
        
        
        # seg_query = self.seg_query.repeat(b, 1, 1)
        # cls_query = self.cls_query.repeat(b, 1, 1)
        part_cls_query = self.part_cls_query.repeat(b, 1, 1)
        
        
        
        # # first_query = torch.cat((seg_query, cls_query), dim=1)  # b,2,768
        # tgt_mask = self.generate_tgt_mask (first_query)
        # first_out = self.Decoder(first_query, encoder_out, tgt_mask=tgt_mask)
        cls_out = encoder_out[:, 0,:]
        layerwise_cls_tokens = cls_out
        # cls_score=self.classifier(cls_out)
        
        # print(segmentation.shape)
            # print(layerwise_part_tokens)

        feat = self.bottleneck(cls_out)

        if self.training:
            
            # seg_out = torch.unsqueeze(first_out[:, 0,:], dim=1)  # b,1,inplane
            # seg_out = seg_out.repeat(1, int(mem_H * mem_W), 1)
            
            feature_map_shape = (b, mem_H, mem_W, self.in_planes)
            
            # seg_out = seg_out.reshape(feature_map_shape)
            feature_map = layerwise_tokens[0][:,1:,:].reshape(feature_map_shape)
            # feature_map = torch.cat((feature_map, seg_out),dim=3).permute(0, 3, 1, 2)
            feature_map =feature_map.permute(0, 3, 1, 2).detach()
            # feature_map=self.upsample(feature_map)
            
            # feature_map = self.upsample(feature_map.detach())
            seg_map=feature_map
            seg_prob=self.cls_seg(seg_map)
            seg_model=self.kmean
            # seg_mask=seg_model.fit_predict(seg_prob.reshape(b,-1,self.prob_depth).detach().cpu()).cuda().reshape(b,mem_H,mem_W).detach()
            seg_mask = torch.argmax(seg_prob, dim=1).detach()
            # print(seg_mask)
            segmentation = [(seg_mask == i) for i in range( self.nb_sem + 1)]  # [32,256,128]*3
            segmentation_pic=segmentation
            segmentation=segmentation[1:]
            segmentation = torch.stack(segmentation, dim=1).reshape(-1,int( self.input_H/16),int( self.input_W/16))
            
            stacked_x = torch.unsqueeze(x, dim=1).repeat(1, self.nb_sem, 1, 1, 1).reshape(-1, 3, self.input_H, self.input_W)
            part_encoder_shape = (b, self.nb_sem, mem_H, mem_W, self.in_planes)
            part_encoder_out = self.base(stacked_x, segmentation)[0] [:,:int(self.nb_sem+1),:] # .reshape(part_encoder_shape)  # 32*3,16,8,768
            
            # sec_out = self.Decoder(sec_query, part_encoder_out, tgt_mask=tgt_mask).reshape(b, self.nb_sem, self.nb_sem, self.in_planes)  # 32,3,3,768
            part_feat_list=[part_encoder_out[:, i,:] for i in range(1,self.nb_sem+1)]
            layerwise_part_tokens = part_feat_list   #torch.stack([sec_out[:, i, i,:] for i in range(self.nb_sem)]       , dim=1)  # 32,3,768

            cls_score = self.classifier(feat)
            grl_feat = self.grl(feat)
            d_score = self.D_classifier(grl_feat)
            part_cls_score = [self.part_classifier(layerwise_part_tokens[i]) for i in range(1,self.nb_sem)]
            
            ###randon img reading
            # ran_imgclass=random.choice(self.random_imgclass_list)
            # ran_img_list=os.listdir(os.path.join('data/datasets/Linnaeus/train',ran_imgclass))
            # ran_img=random.sample(ran_img_list,b)
            # ran_list=[]
            # for i in ran_img:
            #     ran_list.append(os.path.join('data/datasets/Linnaeus/train',ran_imgclass,i))
            # ran_img_list=[]
            # for img in ran_list:
            #     ran_img=PIL.Image.open(img)
            #     ran_img=ran_img.resize((h,w), resample=0) 
            #     ran_img=torchvision.transforms.ToTensor()(ran_img).cuda().reshape(c,h,w)
            #     ran_img_list.append(ran_img)
                
            # ran_input=torch.stack(ran_img_list,dim=0)
            
            # ran_img_tokens= self.base(ran_input)[0][:,1:,::]
            
            # ran_img_tokens=self.upsample(ran_img_tokens.detach())
            
            # ran_img_tokens=ran_img_tokens.reshape(b,h,w,256).permute(0, 3, 1, 2).detach()
            # ran_img_prob=self.cls_seg(ran_img_tokens)
            seg_info = {
                'seg_map':seg_map,
                'seg_mask':seg_mask,
                'seg_prob' : seg_prob,
                'ran_img_map':seg_prob,
                'ran_img_prob':seg_prob
            }
            
            if randint(0,300)==1:
                color_map=np.array([[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255]])
                image = np.zeros((256, 128, 3), dtype=np.uint8)
                
                for i in range(256):
                    for j in range(128):
                        color_index = seg_mask[0][int(i/16),int(j/16)] # Subtract 1 because indices start from 0
                        image[i, j] = color_map[color_index]
                input=x[0,::].permute(1,2,0).cpu().detach().numpy()
                input= (input-np.min(input))/(np.max(input)-np.min(input))*255
                image=image*0.2+input*0.8
                # plt.imshow(image)
                image = np.uint8(image)
                image = Image.fromarray(image)
                image.save(os.path.join('exp_2',f"saved{self.save_t}.png"))
                self.save_t+=1
                print('saved img')
                
                # for i in range(self.nb_sem+1):

                    


                #     # seg_masks =segmentation_pic[i][0,::].cpu().detach().numpy().astype(np.int16)   
                #     seg_masks =segmentation_pic[i][0,::].cpu().detach().numpy().astype(np.int16)   ##
                #     seg_masks*= 255
                #     seg_masks =np.uint8(seg_masks)
                #     seg_masks= np.stack([seg_masks,seg_masks,seg_masks],axis=-1) 
                    
                #     # image = Image.fromarray(seg_masks)
                #     # image.save(os.path.join('exp',f"seg_masks{i}_saved{self.save_t}.png"))
                # norm=torch.functional.norm(seg_map[0,::].permute(1,2,0), dim=2).cpu().detach().numpy()
                # # print(norm.shape)    16,8
                # norm=(norm-np.min(norm))/(np.max(norm)-np.min(norm)  )*255
                
                # norm =np.uint8(norm)
                
                # norm= np.stack([norm,norm,norm],axis=-1) 
                # image = Image.fromarray(norm)
                
                # # image.save(os.path.join('exp',f"norm{0}_saved{self.save_t}.png"))
                # self.save_t+=1
                # input=x[0,::].permute(1,2,0).cpu().detach().numpy()
                # input= (input-np.min(input))/(np.max(input)-np.min(input))*255
                # input = np.uint8(input)
                # image = Image.fromarray(input)
                # # image.save(f"seg_img.png")
                # print('saved img')
            return cls_score, layerwise_cls_tokens, part_feat_list, d_score, seg_info, part_cls_score
        else:
            return feat if self.neck_feat == 'after' else layerwise_cls_tokens

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:  # drop classifier
                continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading trained model from {}'.format(trained_path))

    def generate_tgt_mask(self, tgt):
        b, seq_len, c = tgt.shape
        mask = torch.ones((seq_len, seq_len)).bool().cuda()
        mask = mask.fill_diagonal_(0)
        mask
        # mask=torch.unsqueeze(mask,0)
        # mask=mask.repeat(b,1,1)
        return mask

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('PAT.train')
        logger.info("Number of parameter: %.2fM" % (total / 1e6))       
        
#################


__factory_T_type = {
    'vit_large_patch16_224_TransReID': vit_large_patch16_224_TransReID,
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_base_patch32_224_TransReID': vit_base_patch32_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID,
    'deit_tiny_patch16_224_TransReID': deit_tiny_patch16_224_TransReID,
}

__factory_LAT_type = {
    'vit_large_patch16_224_TransReID': part_attention_vit_large,
    'vit_base_patch16_224_TransReID': part_attention_vit_base,
    'vit_base_patch32_224_TransReID': part_attention_vit_base_p32,
    'deit_base_patch16_224_TransReID': part_attention_vit_base,
    'vit_small_patch16_224_TransReID': part_attention_vit_small,
    'deit_small_patch16_224_TransReID': part_attention_deit_small,
    'deit_tiny_patch16_224_TransReID': part_attention_deit_tiny,
}


class GradientReversalfunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


class deconv2d_bn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=2, strides=2):
        super(deconv2d_bn, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
                                        kernel_size=kernel_size,
                                       stride=strides, bias=True)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x),0.1)
        return out


class GradientReversalLayer(torch.nn.Module):

    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def forward(self, x):
        return GradientReversalfunc.apply(x)


def make_model(cfg, modelname, num_class, sd_flag=False, head_flag=False, camera_num=None, view_num=None, nb_domain=None):
    if modelname == 'ctc_vit':
        model = build_ctc_vit(num_class, cfg, __factory_LAT_type, nb_domain=nb_domain)
        print('===========building ctc_vit===========')
    elif modelname == 'part_attention_vit':
        model = build_part_attention_vit(num_class, cfg, __factory_LAT_type)
        print('===========building our part attention vit===========')
    else:
        model = Backbone(modelname, num_class, cfg)
        print('===========building ResNet===========')
    # ## count params
    model.compute_num_params()
    return model
