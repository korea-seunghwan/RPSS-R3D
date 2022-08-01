import torch
from torch.utils.data import Dataset
import transforms
import pandas as pd
import numpy as np

class RPSDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform 

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        """ get a video and its label """
        file_name = self.dataframe.iloc[index].file_name
        # video_raw = self.dataframe.iloc[index].video_raw
        joint = self.dataframe.iloc[index].joint
        video = self.dataframe.iloc[index].video
        optical = self.dataframe.iloc[index].optical_flow
        # bone = self.dataframe.iloc[index].bone
        # joint_motion = self.dataframe.iloc[index].joint_motion
        # bone_motion = self.dataframe.iloc[index].bone_motion
        trunk = self.dataframe.iloc[index].trunk
        movement = self.dataframe.iloc[index].movement
        shoulder = self.dataframe.iloc[index].shoulder
        elbow = self.dataframe.iloc[index].elbow
        prehension = self.dataframe.iloc[index].prehension
        global_s = self.dataframe.iloc[index].global_s

        # print(joint)
        joint = torch.from_numpy(np.transpose(np.load(joint), (2, 0, 1)))
        video = torch.from_numpy(np.load(video))
        # video = torch.from_numpy(np.transpose(np.load(video), (3, 0, 1, 2)))
        optical = torch.from_numpy(np.transpose(np.load(optical), (3, 0, 1, 2)))

        # bone = np.load(bone)
        # joint_motion = np.load(joint_motion)
        # bone_motion = np.load(bone_motion)
        

        # print(video.shape)
        if self.transform:
            video = self.transform(video)
            optical = self.transform(optical)

        return file_name, video, trunk, movement, shoulder, elbow, prehension, global_s

class RPSDataset_intuitive(Dataset):
    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        """get a vidieo and its label"""
        file_name = self.dataframe.iloc[index].file_name
        joint = self.dataframe.iloc[index].joint
        video = self.dataframe.iloc[index].video
        optical = self.dataframe.iloc[index].optical_flow
        label = self.dataframe.iloc[index].label

        # joint = np.load(joint)
        # video = np.load(video)
        # optical = np.load(optical)
        
        joint = torch.from_numpy(np.transpose(np.load(joint), (2, 0, 1)))
        video = torch.from_numpy(np.load(video))
        # video = torch.from_numpy(np.transpose(np.load(video), (3, 0, 1, 2)))
        optical = torch.from_numpy(np.transpose(np.load(optical), (3, 0, 1, 2)))

        label = label-1

        if self.transform:
            video = self.transform(video)
            optical = self.transform(optical)

        return file_name, joint, video, optical, label