import clip
import torch

import torchvision.transforms as T

class PretrainedModel:
    def __init__(self, dataloader, model_name='ViT-B/32', device=None):
        self.dataloader = dataloader
        
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                        std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def forward(self, x):
        x = self.transform(x)
        x = self.model(x)
        return x
    
    def extract_features(self, dataloader):
        embeddings = []
        labels = []

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.model.visual.conv1.weight.device)
                
                if x.shape[1] == 1:  # If grayscale, convert to 3 channels
                    x = x.repeat(1, 3, 1, 1)
                
                x = self.transform(x)
                
                features = self.model.encode_image(x)
                embeddings.append(features.cpu())
                labels.append(y.cpu())

        return torch.cat(embeddings), torch.cat(labels)



    
