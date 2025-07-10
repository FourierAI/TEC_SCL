import torch
from torch.utils.data import Dataset, DataLoader
import os
import clip
from PIL import Image
import numpy as np
import clip

class Food101Dataset(Dataset):

    def __init__(self, transform=None, device='cuda:0'):
        self.transform = transform
        self.data = self.__readtext()
        self.labels = self.read_label()
        self.device = device
        # 文本编码器
        self.clip_model, self.preprocess = clip.load('RN50', device="cuda:0")
        self.clip_model = self.clip_model.to(self.device)

    def __getitem__(self, index):
        image_path, sentence, label = self.data[index]
        image_path = os.path.join('./datasets', image_path)
        image = Image.open('./datasets/'+image_path)
        try:   
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()/255.0
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = torch.randn(3, 224, 224)  # Fallback to a random tensor if image loading fails
        image = self.transform(image) if self.transform else image   

        # 文本特征
        tokens = clip.tokenize(sentence, truncate=True).to(self.device)
        with torch.no_grad():
            text_embedding, text_feas = self.clip_model.encode_text(tokens)
        # mask
        mask = tokens != 0

        return image, text_feas, mask, self.label_to_index(label)

    def __len__(self):
        return len(self.data)
    
    def __readtext(self):
        data = []
        file_names = os.listdir('./text')
        for file_name in file_names:
            with open(os.path.join('./text', file_name), 'r') as f:
                for line in f:
                    try:
                        image_path, text = line.strip().split('\t\t\t')
                        label = image_path.split('/')[-2]
                        sentence = text.split("\t<>\t")[0]
                        data.append((image_path, sentence, label))
                    except ValueError:
                        print(f"Skipping line in {file_name}: {line.strip()}")
        print(f"Total data points: {len(data)}")
        return data
    
    def read_label(self):
        labels = []
        with open('./datasets/food-101/meta/classes.txt', 'r') as f:
            for line in f:
                label = line.strip()
                labels.append(label)
        return labels

    def label_to_index(self, label):
        labels = self.labels
        if label in labels:
            return labels.index(label)
        else:
            raise ValueError(f"Label '{label}' not found in dataset labels.")
            
    def index_to_label(self, index):
        labels = self.labels
        if 0 <= index < len(labels):
            return labels[index]
        else:
            raise IndexError(f"Index '{index}' is out of bounds for dataset labels.")
