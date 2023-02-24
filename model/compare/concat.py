import torch.nn as nn
import torch


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, y):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2)  # [B,768,196]
        y = self.proj(y).flatten(2)  # [B,768,196]
        return x, y

class ZSLNet(nn.Module):
    def __init__(self, device='cuda:0', norm_layer=None, embed_dim=768):
        super(ZSLNet, self).__init__()
        self.device = device
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x, y):
        return self.forward_ranking(x, y)

    def forward_ranking(self, x, y):
        loss_cos = torch.zeros(1).to(self.device)  # align
        images_loss_mapping_consistency = torch.zeros(1).to(self.device)  # con
        audio_loss_mapping_consistency = torch.zeros(1).to(self.device)  # con

        fc_v = nn.Sequential(
            nn.Linear(196, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 98)
        )

        fc_a = nn.Sequential(
            nn.Linear(196, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 98)
        )

        fc_v = fc_v.to(self.device)
        fc_a = fc_a.to(self.device)

        visual_feats = fc_v(x) 
        audio_feats = fc_a(y)  

        if True: 
            images_mapped_sim = self.sim_score(visual_feats, visual_feats.detach())
            images_orig_sim = self.sim_score(x, x)
            images_loss_mapping_consistency = torch.abs(images_orig_sim - images_mapped_sim).mean()

        if True:  
            audio_mapped_sim = self.sim_score(audio_feats, audio_feats.detach())
            audio_orig_sim = self.sim_score(y, y)
            audio_loss_mapping_consistency = torch.abs(audio_orig_sim - audio_mapped_sim).mean()


        loss_cos = self.CosineLoss(visual_feats, audio_feats)

        a = torch.ones(1).to(self.device)
        b = torch.ones(1).to(self.device)
        c = torch.ones(1).to(self.device)

        feats = torch.cat((visual_feats, audio_feats), dim=2)  # [b,768,98]+[b,768,98]=[b,768,196]
        feats = feats.transpose(1, 2)  # [b,768,196]
        feats = self.norm(feats)
        # feats=visual_feats+audio_feats
        return loss_cos, feats  # [b,c,hw]


    def sim_score(self, a, b):
        a_norm = a / (1e-6 + a.norm(dim=-1)).unsqueeze(2) 
        b_norm = b / (1e-6 + b.norm(dim=-1)).unsqueeze(2)
        score = (torch.matmul(a_norm, b_norm.transpose(1, 2)))
        return score

    def CosineLoss(self, t_emb, v_emb):
        a_norm = v_emb / (1e-6 + v_emb.norm(dim=-1)).unsqueeze(2)
        b_norm = t_emb / (1e-6 + t_emb.norm(dim=-1)).unsqueeze(2)
        loss = 1 - torch.mean(torch.diagonal(torch.matmul(a_norm, b_norm.transpose(1, 2)), 0))
        return loss


class castModel(nn.Module):
    def __init__(self, Embed_layer=PatchEmbed, Cast=ZSLNet):
        super(castModel, self).__init__()
        self.PatchEmbed = Embed_layer()
        self.ZSLNet = Cast()

    def forward(self, x, y):
        a, b = self.PatchEmbed(x, y)
        a, castLoss = self.ZSLNet(a, b)
        return a, castLoss
