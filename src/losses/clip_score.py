import torch
import clip


class CLIPScore(torch.nn.Module):
    def __init__(self, stylegan_size, clip_model, device):
        super().__init__()
        self.model = clip_model
        self.upsample = torch.nn.Upsample(scale_factor=7).to(device)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=stylegan_size // 32).to(device)
        self.device = device
        
        self.values = []
        self.batch_sizes = []
        
    def forward(self, image, text, avg: bool = True):
        image = image.to(self.device)
        text = torch.cat([clip.tokenize(text)]).to(self.device)
        image = self.avg_pool(self.upsample(image))
        
        image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(text)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).float()
        
        if avg:
            return (image_features @ text_features.T).mean()
        
        return (image_features @ text_features.T).sum(), image.shape[0]

    def update(self, image, text):
        score, count = self.forward(image, text, avg=False)
        self.values.append(score.detach().cpu())
        self.batch_sizes.append(count)
    
    def compute(self):
        return sum(self.values) / sum(self.batch_sizes)
    
    def reset(self):
        self.values.clear()
        self.batch_sizes.clear()
