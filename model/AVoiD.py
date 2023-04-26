import copy
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
from MMD import MMD, PatchEmbed, Encoder_layer

class AVoiD(nn.Module):
    def __init__(self, args, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=5, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, device='cuda:0'):

        super(AVoiD, self).__init__()
        self.dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_ratio = drop_ratio
        self.attn_drop_ratio = attn_drop_ratio
        self.act_layer = act_layer
        self.depth = depth
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed_video = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c,
                                             embed_dim=embed_dim)
        self.patch_embed_audio = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c,
                                             embed_dim=embed_dim)
        num_patches = self.patch_embed_video.num_patches

        self.cls_token_video = nn.Parameter(torch.zeros(1, 1, 768))
        self.cls_token_audio = nn.Parameter(torch.zeros(1, 1, 768))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # position
        self.pos_embed_video = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_embed_audio = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop_video = nn.Dropout(p=drop_ratio)
        self.pos_drop_audio = nn.Dropout(p=drop_ratio)
        # time
        self.time_embed_video = nn.Parameter(torch.zeros(1, embed_dim))
        self.time_embed_audio = nn.Parameter(torch.zeros(1, embed_dim))
        self.time_drop_video = nn.Dropout(p=drop_ratio)
        self.time_drop_audio = nn.Dropout(p=drop_ratio)

        self.block = nn.ModuleList()
        for _ in range(depth - 1):
            layer = MMD(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, act_layer=act_layer)
            self.block.append(copy.deepcopy(layer))

        self.last_block = Encoder_layer(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                        qk_scale=qk_scale,
                                        drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio)

        # self.part_select = Search()

        self.video_encoder = nn.Sequential(*[
            Encoder_layer(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio)
            for _ in range(6)
        ])

        self.audio_encoder = nn.Sequential(*[
            Encoder_layer(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio)
            for _ in range(6)
        ])
        self.norm = norm_layer(embed_dim)
        self.av_fc = nn.Linear(embed_dim * 2, embed_dim)
        self.fc = nn.Linear(embed_dim * 3, embed_dim)
        # Select
        # self.Select = Select(bs=args.batch_size, device=args.device, embed_dim=embed_dim,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                             drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio)

        # if representation_size and not distilled:
            # self.has_logits = True
            # self.num_features = representation_size
            # self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        # else:
            # self.has_logits = False
            # self.pre_logits = nn.Identity()

        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # self.head_dist = None
        # if distilled:
            # self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed_video, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_audio, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token_video, std=0.02)
        nn.init.trunc_normal_(self.cls_token_audio, std=0.02)
        self.apply(_init_vit_weights)
        self.device = args.device

        self.w1 = torch.nn.Parameter(torch.ones(1)).to(device)
        self.w2 = torch.nn.Parameter(torch.ones(1)).to(device)
        self.w3 = torch.nn.Parameter(torch.ones(1)).to(device)

    def forward_features(self, video, audio):
        x = self.patch_embed_video(video)
        y = self.patch_embed_audio(audio)
        weight_list_v = []
        weight_list_a = []
        cls_token_video = self.cls_token_video.expand(x.shape[0], -1, -1)
        cls_token_audio = self.cls_token_audio.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token_video, x), dim=1)
            y = torch.cat((cls_token_audio, y), dim=1)
        else:
            x = torch.cat((cls_token_video, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
            y = torch.cat((cls_token_audio, self.dist_token.expand(y.shape[0], -1, -1), y), dim=1)
        # position embed
        x = self.pos_drop_video(x + self.pos_embed_video)
        y = self.pos_drop_audio(y + self.pos_embed_audio)
        # time embed
        x = self.time_drop_video(x + self.time_embed_video)
        y = self.time_drop_audio(y + self.time_embed_video)

        x = self.video_encoder(x)
        y = self.audio_encoder(y)

        Encoder_video = x
        Encoder_audio = y

        # cls_v,cls_v
        cls_v = x[:, 0, :]
        cls_a = y[:, 0, :]

        num_heads = self.Select(cls_v, cls_a)

        block = nn.ModuleList()
        for _ in range(self.depth - 1):
            layer = MMD(dim=self.embed_dim, num_heads=num_heads, mlp_ratio=self.mlp_ratio,
                                qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                                drop_ratio=self.drop_ratio, attn_drop_ratio=self.attn_drop_ratio)
            block.append(copy.deepcopy(layer))

        block.to(self.device)

        for b in block:
            x, y, w_v, w_a = b((x, y, Encoder_video, Encoder_audio))  # w:[bs,num_heads,hidden_size,hidden_size]
            weight_list_v.append(w_v)
            weight_list_a.append(w_a)

        xy = self.av_fc(torch.cat((x, y), dim=-1))

        part_num_va, part_inx_va = self.part_select(weight_list_v, weight_list_a)
        part_inx_va = part_inx_va + 1
        parts_va = []
        B, num = part_inx_va.shape
        for i in range(B):
            parts_va.append(xy[i, part_inx_va[i, :]])  # hidden_states[i, part_inx[i,:]]：[B,num_heads]
        parts_va = torch.stack(parts_va).squeeze(1)
        concat_va = torch.cat((xy[:, 0].unsqueeze(1), parts_va), dim=1)
        x = self.last_block(concat_va)
        fusion_cls = x[:, 0]
        last_cls = self.fc(torch.cat((self.w1 * cls_v, self.w2 * fusion_cls, self.w3 * cls_a), -1))
        return last_cls, cls_v, cls_v

    def forward(self, x, y):
        x, cls_v, cls_v = self.forward_features(x, y)
        feats = x
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x, feats, cls_v, cls_v


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def AVoiD_mm(args, num_classes: int = 21843, has_logits: bool = True):
    model = AVoiD(args=args,
                    img_size=224,
                    patch_size=16,
                    embed_dim=768,
                    depth=6,
                    num_heads=12,
                    representation_size=768 if has_logits else None,
                    num_classes=num_classes)
    return model
