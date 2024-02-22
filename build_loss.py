import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .ce_labelSmooth import CrossEntropyLabelSmooth as CE_LS
from torch.nn.modules.loss import CrossEntropyLoss
import random
import torch
import torchvision
feat_dim_dict = {
    'local_attention_vit': 768,
    'vit': 768,
    'resnet18': 512,
    'resnet34': 512
}
def build_seg_loss(cfg):
    def loss_fn(seg_info,cid,pid):
        seg_map=seg_info['seg_map']   #feature map  b,h,w,768
        seg_mask=seg_info['seg_mask'] # pseudo gt b,h,w
        
        seg_prob=seg_info['seg_prob'].permute(0,2,3,1) # output_prob b,h,w,nb_sem+1
        b,h,w,c=seg_prob.shape
        loss_list=[]
        ##local consistency
        local_loss_list=[]
        for i in [-1,0,1]:
            for j in [-1,0-1]:
                if i==j and  i==0 : continue
                aff=torchvision.transforms.functional.affine(seg_map,0,[i,j], scale=1, shear=0)
                diff=torch.functional.norm(seg_map-aff,dim=3).mean()
                local_loss_list.append(diff)
        LOCAL_LOSS=torch.stack(local_loss_list).mean()
        
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
        
        SEG_COS_LOSS=torch.stack(sem_cons_loss_list).mean()
        ##background grouping
        background_loss_list=[]
        for i in range(seg_map.shape[0]):
            for j in range(seg_map.shape[0]):
                if i==j: continue
                elif cid[i]==cid[j]:
                    loss=torch.functional.norm(seg_map[i,::]-seg_map[j,::],dim=2).mean()
                    background_loss_list.append(loss)
        BACK_GRU_LOSS=torch.stack(background_loss_list).mean()
        
        
        
        ##background determined
        background_det_list=[]
        background_det_loss=CrossEntropyLoss()
        
        det_list=[seg_prob[:,0,:,:], seg_prob[:,:,0,:],seg_prob[:,-1,:,:],seg_prob[:,:,-1,:]]
        for i in det_list:
            gt=torch.zeros((int(i.shape[0]*i.shape[1])) ).long().cuda()
            background_det_list.append(background_det_loss(i.reshape(-1,c),gt))
        # det_list=[seg_prob[:,8,:,:], seg_prob[:,:,8,:],seg_prob[:,-8,:,:],seg_prob[:,:,-8,:]]
        # for i in det_list:
        #     gt=torch.ones((int(i.shape[0]*i.shape[1])) ).long().cuda()
        #     background_det_list.append(background_det_loss(i.reshape(-1,c),gt))
        BACK_DET_LOSS=torch.stack(background_det_list).mean()*0
        
        
        return (SEG_COS_LOSS+LOCAL_LOSS+BACK_GRU_LOSS+BACK_DET_LOSS)*0.1
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


