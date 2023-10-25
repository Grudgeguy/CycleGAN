import random,os,torch, numpy as np
import torch.nn as nn
import config
import copy

def save_checkpoint(model,optimizer,filename="check.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict" :model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint,filename)

def load_checkpoint(checkpoint,model,optimizer,lr):
    print("=> Loading Model")
    checkpoint = torch.load(checkpoint,map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for params in optimizer.param_groups:
        params["lr"]=lr
