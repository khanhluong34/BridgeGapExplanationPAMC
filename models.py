import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights #
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchsummary import summary 
from backbone.convnext import convnext_base as convnext_base_22
class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    
    def forward(self, feature_map):
        return F.adaptive_avg_pool2d(feature_map, 1).squeeze(-1).squeeze(-1)

class ImageClassifier(torch.nn.Module):
    def __init__(self, P):
        super(ImageClassifier, self).__init__()
        
        self.arch = P['arch']
        if P['dataset'] == 'OPENIMAGES':
            feature_extractor = torchvision.models.resnet101(pretrained=P['use_pretrained'])
        else:
            if self.arch == 'convnext-tiny': #
                feature_extractor = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)#
            elif self.arch == 'convnext-base': 
                feature_extractor = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
                # feature_extractor = convnext_base_22(pretrained=P['use_pretrained'], in_22k=True)
            else: #
                feature_extractor = torchvision.models.resnet50(pretrained=P['use_pretrained'])

        # if self.arch == 'convnext-base': 
        #     feature_extractor.norm = nn.Identity()
        #     feature_extractor.head = nn.Identity()
        # else: 
        #     feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-2])
        feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-2])


        if P['freeze_feature_extractor']:
            for param in feature_extractor.parameters():    
                param.requires_grad = False
        else:
            for param in feature_extractor.parameters():
                param.requires_grad = True

        self.feature_extractor = feature_extractor
        # self.layernorm = nn.LayerNorm((P['feat_dim'],), eps=1e-06, elementwise_affine=True) #
        self.avgpool = GlobalAvgPool2d()
        self.onebyone_conv = nn.Conv2d(P['feat_dim'], P['num_classes'], 1)
        self.alpha = P['alpha']

    def unfreeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        feats = self.feature_extractor(x)

        # if self.arch == "convnext-base": 
        #     CAM = self.onebyone_conv(torch.squeeze(feats, 2))
        # else: 
        #     CAM = self.onebyone_conv(feats)
        CAM = self.onebyone_conv(feats)

        CAM = torch.where(CAM > 0, CAM * self.alpha, CAM) # BoostLU operation
        logits = F.adaptive_avg_pool2d(CAM, 1).squeeze(-1).squeeze(-1)
        return logits

if __name__ == "__main__": 
    # feature_extractor = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).to('cuda')
    # feature_extractor = convnext_base_22(pretrained=True, in_22k=True).to('cuda')
    feature_extractor = convnext_base_22().to('cuda')
    # convnext1k = convnext_base()
    # feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-2])
    # feature_extractor = feature_extractor.backbone
    # print(backbone)
    # feature_extractor.norm = nn.Conv2d(1024, 41, 1)
    # feature_extractor.head = nn.Identity()
    # print(summary(feature_extractor, (3, 224, 224)))
    print(*list(feature_extractor.children()))
    # print(summary(convnext1k.to('cuda'), (3, 224, 224)))

