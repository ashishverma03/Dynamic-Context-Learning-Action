from os import name
import argparse
import pdb
from tabnanny import check
from matplotlib.pyplot import axis
from numpy import mod
#from parso import parse
import numpy as np
# from cv2 import transform
import torch
import logging
import os
import time
from myutils import encode_labels,plot_history
from Network.training import test_model

from torch.utils.data import DataLoader
from Loader import metric,test_voc_fetcher


from Loader.transformations import DCNDefaultTrainTransform,DCNDefaultValTransform
import torchvision

from Network import network_res_lstm_412


num = 4121

if __name__ == '__main__':
    # Initialize cuda parameters
    use_cuda = torch.cuda.is_available()
    use_cuda = 1
    np.random.seed(1)
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")


    model_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models")
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)


    # Dataloader
    val_dataset = test_voc_fetcher.VOCAction(split='test', load_box=True,transform=DCNDefaultValTransform(600,1000))
    val_data = DataLoader(val_dataset, batch_size=1, shuffle=False,drop_last=False)
    eval_metric = metric.VOCMultiClsMApMetric(class_names=val_dataset.classes, ignore_label=-1, voc_action_type=True)


    #Model
    net = network_res_lstm_412.Custom_Model(pretrained=True,device=device).to(device)
    
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    #Load the weights
    weights_file_path =  os.path.join(model_dir, "model-{}.pth".format(num))
    if os.path.isfile(weights_file_path):
        print("Loading best weights")
        # pdb.set_trace()
        checkpoint = torch.load(weights_file_path)
        net.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])


    #Training
    val_loss_list = []
    val_ap_list = []
       
        
    test_model(net, device, criterion, val_data, eval_metric) 

    torch.cuda.empty_cache()
 