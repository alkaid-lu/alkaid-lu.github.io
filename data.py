import torch
import torch.nn as nn
from torch.utils.data import Dataset

SMILE = 0
DENSITY = 1
CALORICITY = 2
MELTING = 3


class Monecular(Dataset):
    def __init__(self, datapath):
        with open(datapath) as fd:
            self.samples = fd.readlines()[1:]
        
        self.vocab = ['C', 'H', '1', '2', '3', '4', '=', '(', ')', '[', ']', '@']
        self.c2inx={}
        for i, c in enumerate(self.vocab):
            self.c2inx[c] = i
               
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx].strip()
        sample = sample.split(',')
                    
        smile = sample[SMILE]
        length=len(smile)
        feature_idx_list = []
        for c in smile:
            feature_idx_list.append(self.c2inx[c])
        feature = torch.nn.functional.one_hot(torch.tensor(feature_idx_list), num_classes=12)
        
#class torch.nn.ZeroPad2d():
        pad=nn.ZeroPad2d((0,0,0,(40-length)))      
        feature_pad = torch.tensor(pad(feature),dtype=torch.float32)
                            
        density=float(sample[DENSITY])
        density = torch.tensor(density)
        caloricity=float(sample[CALORICITY])
        caloricity = torch.tensor(caloricity)
        melting=float(sample[MELTING])
        melting = torch.tensor(melting)
        
        return feature, (density, caloricity, melting)

