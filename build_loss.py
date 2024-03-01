import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .ce_labelSmooth import CrossEntropyLabelSmooth as CE_LS
from torch.nn.modules.loss import CrossEntropyLoss
import random
import torch
import torchvision
from torch_kmeans  import KMeans
import numpy as np
from PIL import Image
from torch_kmeans.utils.distances import (
    BaseDistance,
    CosineSimilarity,
    DotProductSimilarity,
    LpDistance,
)
from .hard_tri_loss import HardTripletLoss
import sys,os
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
feat_dim_dict = {
    'local_attention_vit': 768,
    'vit': 768,
    'resnet18': 512,
    'resnet34': 512
}
def cor_loss(cfg,seg_info,pid):
    cor_list=[]
    seg_map=seg_info['seg_map'] .permute(0,2,3,1)  #feature map  b,h,w,768
    seg_mask=seg_info['seg_mask'] # pseudo gt b,h,w
    seg_prob=seg_info['seg_prob'].permute(0,2,3,1) # output_prob b,h,w,nb_sem+1
    b,h,w,c=seg_prob.shape
    depth=seg_map.shape[-1]
    
    lambda_knn=cfg.MODEL.LAMBDA_KNN
    lambda_self=cfg.MODEL.LAMBDA_SELF
    lambda_ran=cfg.MODEL.LAMBDA_RAN
    b_knn=cfg.MODEL.B_KNN
    b_self=cfg.MODEL.B_SELF
    b_ran=cfg.MODEL.B_RAN
    
    for i in range(b):
        f_self=seg_map[i,::]
        p_self=seg_prob[i,::]
        p_i=pid[i]   #person i 
        ##  sample same id
        same_id_sample_list=[]
        dif_id_sample_list=[]
        
        for j in range(b):
            if i == j: continue
            if pid[j]==  p_i:
                same_id_sample_list.append(j)
            else:
                dif_id_sample_list.append(j)
        
        ##samepling
        same_id=random.choice(same_id_sample_list)
        dif_id=random.choice(dif_id_sample_list)
        
        f_knn=seg_map[same_id,::]
        p_knn=seg_prob[same_id,::]
        f_ran=seg_map[dif_id,::]
        p_ran=seg_prob[dif_id,::]
        
        f_knn_norm=torch.nn.functional.normalize(f_knn,dim=2)
        f_ran_norm=torch.nn.functional.normalize(f_self,dim=2)
        f_self_norm=torch.nn.functional.normalize(f_ran,dim=2)
        p_knn_norm=torch.nn.functional.normalize(p_knn,dim=2)
        p_ran_norm=torch.nn.functional.normalize(p_ran,dim=2)
        p_self_norm=torch.nn.functional.normalize(p_self,dim=2)
        
        
        F_knn=torch.einsum('ijc,hwc->ijhw',f_self_norm, f_knn_norm)
        F_self=torch.einsum('ijc,hwc->ijhw',f_self_norm, f_self_norm)
        F_ran=torch.einsum('ijc,hwc->ijhw',f_self_norm, f_ran_norm)
        S_knn=torch.einsum('ijc,hwc->ijhw',p_self_norm, p_knn_norm)
        S_self=torch.einsum('ijc,hwc->ijhw',p_self_norm, p_self_norm)
        S_ran=torch.einsum('ijc,hwc->ijhw',p_self_norm, p_ran_norm)
        
        F_sc_knn=F_knn-torch.mean(F_knn,dim=(2,3))
        F_sc_self=F_knn-torch.mean(F_self,dim=(2,3))
        F_sc_ran=F_knn-torch.mean(F_ran,dim=(2,3))
        
        S_knn[S_knn<0]=0
        S_self[S_self<0]=0
        S_ran[S_ran<0]=0
        
        knn_loss=-torch.sum(torch.multiply((F_sc_knn-b_knn),S_knn))
        self_loss=-torch.sum(torch.multiply((F_sc_self-b_self),S_self))
        ran_loss=-torch.sum(torch.multiply((F_sc_ran-b_ran),S_ran) )
        
        loss=knn_loss*lambda_knn+self_loss*lambda_self+ran_loss*lambda_ran
        cor_list.append(loss)
    return      torch.stack(cor_list).mean()  
        
        
        
    
def build_seg_loss(cfg):
    def loss_fn(seg_info,cid,pid):
        seg_map=seg_info['seg_map'] .permute(0,2,3,1)  #feature map  b,h,w,768
        seg_mask=seg_info['seg_mask'] # pseudo gt b,h,w
        
        seg_prob=seg_info['seg_prob'].permute(0,2,3,1) # output_prob b,h,w,nb_sem+1
        b,h,w,c=seg_prob.shape
        depth=seg_map.shape[-1]
        loss_list=[]
        cd_loss=cor_loss(cfg,seg_info,pid)
        # p_tri_loss=HardTripletLoss()
        # ce_loss=CrossEntropyLoss()
        # ##setting pseudo gt
        # # print(torch.norm(seg_map,dim=3))
        # max_fc=0
        # done_pid=[]
        # tri_list=[]
        # ce_list=[]
        # for i in range(seg_map.shape[0]):
        #     same_id_list=[]
        #     same_prob_list=[]
        #     id_list=[]
        #     if pid[i] in done_pid: continue
        #     for j in range(seg_map.shape[0]):
        #         # if i==j: continue
        #         # if pid[i]==pid[j]:
        #             same_id_list.append(seg_map[j,::].cpu())
        #             same_prob_list.append(seg_prob[j,::].cpu())
                    
        #             id_list.append(j)
        #             if len(id_list)==2:
        #                 break
        #     # blockPrint()
        #     done_pid.append(pid[i])
        #     same_id=torch.cat(same_id_list,dim=0)
        #     same_prob=torch.cat(same_prob_list,dim=0)
        #     id_list=np.array(id_list).reshape(-1,1) .repeat(128,axis=1).reshape(-1)
        #     id_list[id_list>np.mean(id_list)]=1
        #     id_list[id_list!=1]=0
        #     print(id_list)
            
        #     # print(torch.norm(same_id.reshape(-1,depth),dim=1).reshape(-1,1).shape)
        #     # print(same_id.shape)
        #     norm=torch.norm(same_id,dim=2)
        #     # print(norm.shape)
        #     tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
        #     X_tsne = tsne.fit_transform(same_id.reshape(-1,depth).detach().cpu().numpy())
        #     x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        #     X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
        #     plt.figure(figsize=(8, 8))
        #     for i in range(X_norm.shape[0]):
        #         plt.text(X_norm[i, 0], X_norm[i, 1], 'o', color=plt.cm.Set1(id_list[i]), 
        #                 fontdict={'weight': 'bold', 'size': 9})
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.show()
        #     # norm=torch.div(torch.subtract( norm,torch.mean(norm,dim=0)),torch.std(norm,dim=0)) .reshape(-1,h*w,1)
            
        #     # outlier=torch.nonzero(norm>1)
        #     # if outlier.shape[0]>=300:
        #     #     print(outlier.shape , 'in',norm.shape )
        #     # norm[norm>1]=1
        #     # norm=torch.cat((norm,norm),dim=2)
        #     # model = KMeans(n_clusters=2,distance=LpDistance,verbose=False)
        #     # result = model(norm)
        #     # centers=result.centers
            
        #     # labels=result.labels.reshape(-1)
        #     # # print(result.labels.reshape(-1,h,w).shape)
        #     # seg_masks =result.labels.reshape(-1,h,w)[0,:,:].cpu().detach().numpy().astype(np.int16)   ##
        #     # seg_masks*= 255
        #     # seg_masks =np.uint8(seg_masks)
        #     # seg_masks= np.stack([seg_masks,seg_masks,seg_masks],axis=-1) 
            
        #     # image = Image.fromarray(seg_masks)
        #     # image.save(os.path.join('exp',f"label{0}_saved{4}.png"))
            
        #     # print(centers.shape)
        #     # cluster_ids_x, cluster_centers = kmeans(
        #     # X=torch.norm(same_id.reshape(-1,depth),dim=1).reshape(-1,1), num_clusters=2, distance='euclidean', device='cpu',tol=0.0001)
        #     # ##seperate bg fg
        #     # bg_id=int(centers[0,1,1]<centers[0,0,0])
        #     # fg_id=int(centers[0,1,1]>=centers[0,0,0])
        #     # # print(same_prob.reshape(-1,c) [cluster_ids_x==bg_id])
        #     # ce_list.append(ce_loss(same_prob.reshape(-1,c) [labels==bg_id].cpu(),labels[labels==bg_id].cpu()))
            
            
            
        #     #bg
        #     # fg_feature=same_id.reshape(-1,depth)[labels==fg_id]
        #     # fg_prob=same_prob.reshape(-1,c)[labels==fg_id]
        #     fg_feature=torch.nn.functional.normalize(same_id.reshape(-1,depth), p=2, dim=1).reshape(1,-1,depth)
        #     model = KMeans(n_clusters=c,distance=CosineSimilarity,verbose=False)
        #     # print(fg_feature.reshape(1,-1,depth).shape)
        #     labels = model.fit_predict(fg_feature.reshape(1,-1,depth))
        #     labels=labels.reshape(-1).cpu()
        #     # cluster_ids_x, cluster_centers = kmeans(
        #     # X=fg_feature.reshape(-1,depth), num_clusters=c-1, distance='cosine', device='cpu',tol=0.0001)
        #     #  ##we need to consist among all img
            
        #     ce_list.append(ce_loss(same_prob.reshape(-1,c) .cpu(),labels))
        #     enablePrint()
        #     # tri_list.append(p_tri_loss(same_id.reshape(-1,depth).cuda(),cluster_ids_x.cuda()))
            
        P_LOSS=cd_loss #torch.stack(ce_list,dim=0).mean().cpu()   #torch.stack(tri_list,dim=0).mean().cpu()+#hape(-1,depth).cpu(),cluster_ids_x.cpu())[0]
        
        ##local consistency
        local_loss_list=[]
        for i in [-1,0,1]:
            for j in [-1,0-1]:
                if i==j and  i==0 : continue
                aff=torchvision.transforms.functional.affine(seg_map,0,[i,j], scale=1, shear=0)
                diff=torch.functional.norm(seg_map-aff,dim=3).mean()
                local_loss_list.append(diff)
        LOCAL_LOSS=torch.stack(local_loss_list).mean()*0

        ##semantic_consistency
        sem_cons_loss=CrossEntropyLoss()
        sem_cons_loss_list=[]
        for i in range(seg_map.shape[0]):
            for j in range(seg_map.shape[0]):
                if i==j: continue
                elif pid[i]==pid[j]:
                    
                    p_gt=seg_mask[i,::].reshape(-1)
                    logit=seg_prob[j,::]
                    logit=logit.reshape(-1,logit.shape[-1])
                    # print(logit.shape)
                    sem_cons_loss_list.append(sem_cons_loss(logit,p_gt))
        
        SEG_COS_LOSS=torch.stack(sem_cons_loss_list).mean()*0
        ##background grouping
        background_loss_list=[]
        for i in range(seg_map.shape[0]):
            for j in range(seg_map.shape[0]):
                if i==j: continue
                elif cid[i]==cid[j]:
                    loss=torch.functional.norm(seg_map[i,::]-seg_map[j,::],dim=2).mean()
                    background_loss_list.append(loss)
        BACK_GRU_LOSS=torch.stack(background_loss_list).mean()*0
        
        
        
        ##background determined
        background_det_list=[]
        background_det_loss=CrossEntropyLoss()
        
        det_list=[seg_prob[:,0,:,:], seg_prob[:,:,0,:],seg_prob[:,-1,:,:],seg_prob[:,:,-1,:]]
        for i in det_list:
            gt=torch.zeros((int(i.shape[0]*i.shape[1])) ).long().cuda()
            background_det_list.append(background_det_loss(i.reshape(-1,c),gt))
        det_list=[seg_prob[:,2,:,:], seg_prob[:,:,2,:],seg_prob[:,-2,:,:],seg_prob[:,:,-2,:]]
        for i in det_list:
            gt=torch.ones((int(i.shape[0]*i.shape[1])) ).long().cuda()
            background_det_list.append(background_det_loss(i.reshape(-1,c),gt))
        # det_list=[seg_prob[:,8,:,:], seg_prob[:,:,8,:],seg_prob[:,-8,:,:],seg_prob[:,:,-8,:]]
        # for i in det_list:
        #     gt=torch.ones((int(i.shape[0]*i.shape[1])) ).long().cuda()
        #     background_det_list.append(background_det_loss(i.reshape(-1,c),gt))
        BACK_DET_LOSS=torch.stack(background_det_list).mean()*0
        
        
        return (SEG_COS_LOSS+LOCAL_LOSS+BACK_GRU_LOSS+BACK_DET_LOSS+P_LOSS)
    return loss_fn
        
def build_loss(cfg, num_classes,nb_domain):
    name = cfg.MODEL.NAME
    
    sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.NAME not in feat_dim_dict.keys():
        feat_dim = 2048
    else:
        feat_dim = feat_dim_dict[cfg.MODEL.NAME]
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=False)  # center loss
    if 'cos' in cfg.MODEL.SEM_LOSS_TYPE:
        sem_triplet=TripletLoss()  # triplet loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
        
    if 'crossentropy' in cfg.GRL_LOSS_TYPE :
        grl_loss= CrossEntropyLoss()
        
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        if name == 'local_attention_vit' and cfg.MODEL.PC_LOSS:
            xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        else:
            xent = CE_LS(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax': # softmax loss only
        print('loss_func here11111111111111111')
        
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    # softmax & triplet
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet' or 'GS':
        print('22222222222222222222222222222loss_func here2')
        
        def loss_func(score, feat, target,part_cls_score=None,part_feat=None,gt_domain=None, pred_domain=None, all_posvid=None, soft_label=False, soft_weight=0.1, soft_lambda=0.2,segmentation=None):
            if 'cos' in cfg.MODEL.SEM_LOSS_TYPE and segmentation is not None and part_cls_score is not None:
                nb_sem=cfg.MODEL.NB_SEM
                # part_cls_score=torch.stack(part_cls_score)
                part_feat=torch.cat(part_feat,dim=0)
                b=score.shape[0]
                part_mask_target=[]
                sem_loss_list=[]
                part_xent_list=[]
                for i in range(nb_sem):
                    for j in range(nb_sem):
                        if j>=i: break
                        sem_loss_list.append(torch.nn.functional.cosine_similarity(segmentation[:,i],segmentation[:,j]))
                    part_xent_loss = xent(part_cls_score[i], target) #, all_posvid=all_posvid, soft_label=soft_label,soft_weight=soft_weight, soft_lambda=soft_lambda)
                        
                    part_xent_list.append(part_xent_loss)
                    part_mask_target=part_mask_target+[i]*b
                part_mask_target=torch.tensor(part_mask_target).cuda()
                # print(part_mask_target   ,part_feat.shape )
                SEM_LOSS= torch.abs(segmentation-0.5).mean() *0 +\
                     torch.stack(part_xent_list).mean()*0  +\
                         cfg.MODEL.TRIPLET_LOSS_WEIGHT *   sem_triplet(part_feat, part_mask_target)[0] *0.8
                        
                        # torch.abs(segmentation-0.5).mean()
                
            if gt_domain is not None:
                GRL_LOSS=grl_loss(pred_domain,gt_domain)
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if name == 'local_attention_vit' and cfg.MODEL.PC_LOSS:
                        ID_LOSS = xent(score, target, all_posvid=all_posvid, soft_label=soft_label,soft_weight=soft_weight, soft_lambda=soft_lambda)
                    else:
                        ID_LOSS = xent(score, target)
                else:
                    ID_LOSS = F.cross_entropy(score, target)

                TRI_LOSS = triplet(feat, target)[0]
                # DOMAIN_LOSS = xent(domains, t_domains)
                
                if random.randint(0,100)==80:
                    print(f'showing loss with SEMLOSS:{cfg.MODEL.ID_LOSS_WEIGHT*SEM_LOSS},ID_LOSS:{cfg.MODEL.ID_LOSS_WEIGHT*ID_LOSS}, TRI_LOSS:{  cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS},GRL_LOSS:{cfg.MODEL.ID_LOSS_WEIGHT*GRL_LOSS}')
                # print(ID_LOSS,SEM_LOSS,TRI_LOSS,GRL_LOSS)
                return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS+\
                                 cfg.MODEL.ID_LOSS_WEIGHT *  GRL_LOSS +\
                                     cfg.MODEL.ID_LOSS_WEIGHT * SEM_LOSS
            elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)+\
                            GRL_LOSS
                else:
                    return F.cross_entropy(score, target) + \
                            triplet(feat, target)[0] + \
                            cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target) +\
                                GRL_LOSS
            else:
                print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
                    'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


