import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image

from sentence_transformers import SentenceTransformer

class PretrainedCLIPModel(nn.Module):
    def __init__(self, model_arc='clip-ViT-B-32', device='cuda'):
        super().__init__()
        self.device = device
        self.model_arc = model_arc
        self.embedding_dim = 512
        self.use_fallback = False

        self.clip = SentenceTransformer(model_arc, device=device)
        self.projection = nn.Linear(512, 128)

    def forward(self, x):
        # Convert grayscale to RGB
        if x.shape[1] == 1:
            x_rgb = x.repeat(1, 3, 1, 1)
        else:
            x_rgb = x

        # Normalize to [0, 1]
        if x_rgb.min() < 0:
            x_rgb = (x_rgb + 1) / 2

        x_rgb = x_rgb.to(self.device)

        with torch.no_grad():
            if self.use_fallback:
                z = self.clip.encode_image(x_rgb)
            else:
                z_list = []
                for img_tensor in x_rgb:
                    # Convert tensor to PIL Image
                    img_np = (img_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
                    if img_np.shape[2] == 3:
                        img_pil = Image.fromarray(img_np)
                    else:
                        img_pil = Image.fromarray(img_np[:, :, 0])
                    z_list.append(img_pil)

                # Encode
                z = self.clip.encode(z_list, convert_to_tensor=True)
                z = z.to(self.device)

        z = F.normalize(z, dim=1)

        z_proj = self.projection(z)
        z_proj = F.normalize(z_proj, dim=1)

        return z, z_proj

    def encode(self, x):
        z, z_proj = self(x)
        return z_proj
