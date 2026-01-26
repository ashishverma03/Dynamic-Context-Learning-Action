from matplotlib.pyplot import polar
import torch
import timm
import torch.nn as nn
from zmq import device
# from Network.model import resnet50
from torchvision.models import resnet50
import torchvision
import numpy as np
import pdb
from Fixations.Seq_Embed import FixationSequenceEmbedding,Sequencer
# from Network.relation import HumanObjectRelationModule


class Custom_Model(torch.nn.Module):
    def __init__(self,pretrained,device):
        """
        
        """
        super(Custom_Model, self).__init__()
        # self.resnet = resnet50(pretrained=True)
        self.resnet = timm.create_model('resnet152d', pretrained=True)
        self.feat_dim = 1080
        num_directions = 1
        # print("rngrfireniern: ",list(self.resnet.children()))
        self.resnet1 = torch.nn.Sequential(*(list(self.resnet.children())[0:7]))
        self.resnet2 = torch.nn.Sequential(*(list(self.resnet.children())[7:9]))
        # self.global_avg_pool = torch.nn.AvgPool2d(7,ceil_mode=True)
        # self.model2 = torch.nn.Sequential()
        self.lstm = torch.nn.LSTM(self.feat_dim,self.feat_dim,num_layers=1, batch_first=True)
        self.device = device

        # self.relation = HumanObjectRelationModule()

        # self.global_avg_pool = torch.nn.AvgPool2d(kernel_size=(7,7),stride=2)
        # self.dropout = torch.nn.Dropout(p=0.25)
        self.fc1 = torch.nn.Linear(in_features= 2048, out_features=self.feat_dim, bias=True)
        self.fc2 = torch.nn.Linear(in_features= 2048, out_features=self.feat_dim, bias=True)
        self.fc3 = torch.nn.Linear(in_features= 2048, out_features=self.feat_dim, bias=True)
        self.fc4 = torch.nn.Linear(in_features= 2048, out_features=self.feat_dim, bias=True)
        self.class_predictor = torch.nn.Linear(in_features= self.feat_dim, out_features=40, bias=True)
        self.ctx_class_predictor = torch.nn.Linear(in_features= self.feat_dim, out_features=40, bias=True)
        self.conv = torch.nn.Conv2d( 6*self.feat_dim, self.feat_dim, 1, 1, 0, groups=6)

        nn.init.normal_(self.fc1.weight, mean=0, std=0.01)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.01)
        nn.init.normal_(self.fc3.weight, mean=0, std=0.01)
        nn.init.normal_(self.fc4.weight, mean=0, std=0.01)
        nn.init.normal_( self.class_predictor.weight, mean=0, std=0.01)
        nn.init.normal_(self.ctx_class_predictor.weight, mean=0, std=0.01)
        nn.init.normal_(self.conv.weight, mean=0, std=0.01)

        # nn.init.xavier_normal_(self.fc1.weight)
        # nn.init.xavier_normal_(self.fc2.weight)
        # nn.init.xavier_normal_(self.fc3.weight)
        # nn.init.xavier_normal_(self.fc4.weight)
        # nn.init.xavier_normal_( self.class_predictor.weight)
        # nn.init.xavier_normal_(self.ctx_class_predictor.weight)

        
    def forward(self, x, gt_box = None, obj_box = None, fixations = None):
        """

        """
        
        # pdb.set_trace()
        gt_box = gt_box.reshape(gt_box.shape[1],gt_box.shape[2])
        
        obj_box = obj_box.reshape(obj_box.shape[1],obj_box.shape[2])
        all_rois = torch.row_stack((gt_box,obj_box))
        all_rois = all_rois.reshape(1,all_rois.shape[0],all_rois.shape[1])

        num_scanpaths = np.int(fixations.shape[1]/40)
        num_hum = np.int(gt_box.shape[0])
        num_obj = np.int(obj_box.shape[0])
        
        fixations = fixations.reshape(1,num_scanpaths,40,2)
        # pdb.set_trace()
        # with torch.no_grad():
        feat = self.resnet1(x)

        pooled_feat = torchvision.ops.roi_align(feat, list(gt_box.unsqueeze(0)), output_size=(14, 14), spatial_scale=0.0625,sampling_ratio=2)
        pooled_ctx_feat = torchvision.ops.roi_align(feat, list(obj_box.unsqueeze(0)), output_size=(14, 14), spatial_scale=0.0625,sampling_ratio=2)

        top_feat =  self.resnet2(pooled_feat)
        top_ctx_feat =  self.resnet2(pooled_ctx_feat)



        top_feat = self.fc1(top_feat)




        feat2 = self.resnet2(feat)
        top_ctx_feat = torch.add(top_ctx_feat, feat2)
        top_ctx_feat = self.fc2(top_ctx_feat)

        feat2 = self.fc4(feat2)


        
        top_feat = torch.add(top_feat, feat2)


        output_feat = torch.zeros((num_scanpaths,self.feat_dim)).cuda()
        # output_ctx_feat = torch.zeros((num_scanpaths,num_obj,1024)).cuda()
        for scan_id in range(0,num_scanpaths):
            scanpath = fixations[0][scan_id]
            scanpath = scanpath[scanpath.sum(1)!=0].unsqueeze(0)
            if scanpath is not None:
                all_rois = Sequencer(all_rois,scanpath,self.device)
                all_rois = all_rois.reshape(1,all_rois.shape[0],all_rois.shape[1])
            all_rois_list = list(all_rois)
    
            
            pooled_scpth_feat = torchvision.ops.roi_align(feat, all_rois_list, output_size=(14, 14), spatial_scale=0.0625,sampling_ratio=2)
            top_scpth_feat = self.resnet2(pooled_scpth_feat)

            # top_scpth_feat = self.global_avg_pool(top_scpth_feat)
  
            # top_scpth_feat = top_scpth_feat.flatten(1,3)
            top_scpth_feat = self.fc3(top_scpth_feat)
            # top_scpth_feat = self.dropout(top_scpth_feat)
            top_scpth_feat = top_scpth_feat.unsqueeze(0)
            # pdb.set_trace()
       
            # hidden = (torch.zeros(1, 1, 1024).to(self.device), torch.zeros(1, 1, 1024).to(self.device))
            hidden = (feat2.unsqueeze(0), feat2.unsqueeze(0))
            # hidden = (feat2.unsqueeze(0).repeat(2,1,1), feat2.unsqueeze(0).repeat(2,1,1))

            out, hidden = self.lstm(top_scpth_feat,hidden)

            if isinstance(hidden, tuple):
                hidden, cell = hidden 

            hidden = hidden[-1]

            hidden1 = hidden.squeeze(0)
            output_feat[scan_id,:] =  hidden1


        # # # --------------------------------------------------

        # pred_feat = self.class_predictor(output_feat)
        # pred_ctx_feat = self.ctx_class_predictor(output_ctx_feat)

        # cls_pred = torch.max(pred_feat,0)[0] 
        # ctx_cls_pred = torch.max(pred_ctx_feat,0)[0]

        # cls_pred = cls_pred.unsqueeze(0)
        # ctx_cls_pred = ctx_cls_pred.unsqueeze(0)

        # ctx_cls_pred = torch.max(ctx_cls_pred,1,keepdims=True)
        # cls_pred = torch.add(cls_pred, ctx_cls_pred[0])

        # # # --------------------------------------------------
        # pdb.set_trace()
        out_feat = output_feat.reshape((1,num_scanpaths*self.feat_dim,1,1)) 
        out_feat = self.conv(out_feat).flatten(1,3)

        top_feat = top_feat + out_feat
        top_ctx_feat = top_ctx_feat + out_feat
        cls_pred = self.class_predictor(top_feat)
        ctx_cls_pred = self.ctx_class_predictor(top_ctx_feat)

        # cls_pred = torch.max(pred_feat,0)[0] 
        # ctx_cls_pred = torch.max(pred_ctx_feat,0)[0]

        cls_pred = cls_pred.unsqueeze(0)
        ctx_cls_pred = ctx_cls_pred.unsqueeze(0)

        ctx_cls_pred = torch.max(ctx_cls_pred,1,keepdims=True)
        cls_pred = torch.add(cls_pred, ctx_cls_pred[0])
        
        return cls_pred

