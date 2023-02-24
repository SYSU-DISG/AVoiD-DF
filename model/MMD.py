import torch
import torch.nn as nn

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x)
        x = x.resize_([x.shape[0], x.shape[1], x.shape[2] + 1, x.shape[2] + 1])
        x = self.pooling(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x  # [b,196,768]


class att(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False, scale=None, attn_drop_ratio=0., drop_path_ratio=0.,
                 proj_drop_ratio=0.):
        super(att, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = scale or head_dim ** -0.5
        # q、k、v
        self.q = nn.Linear(dim, dim, bias=bias)
        self.k = nn.Linear(dim, dim, bias=bias)
        self.v = nn.Linear(dim, dim, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.LayerNorm = torch.nn.LayerNorm(dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

    def forward(self, input):
        hidden_states, context = input
        B, N, C = hidden_states.shape
        mixed_query = self.q(hidden_states).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        mixed_key = self.k(context).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        mixed_value = self.v(context).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (mixed_query @ mixed_key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        weight = attn
        attn = self.attn_drop(attn)

        x = (attn @ mixed_value).transpose(1, 2).reshape(B, N, C)  # [b,196,768]
        x = self.proj(x)
        x = self.proj_drop(x)

        x = self.drop_path(x)
        x = self.LayerNorm(x + hidden_states)
        return x, weight


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop_path_ratio=0.,
                 drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.LayerNorm = torch.nn.LayerNorm(in_features)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

    def forward(self, input):
        x = input
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.drop_path(x)
        x = self.LayerNorm(input + x)
        return x


class Encoder_layer(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 act_layer=nn.GELU):
        super(Encoder_layer, self).__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.attention = att(dim, num_heads=num_heads, bias=qkv_bias, scale=qk_scale,
                             attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.fc = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, feats):
        attention_output, _ = self.attention((feats, feats))
        fc_output = self.fc(attention_output)
        return fc_output


class AVF_Decoder(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU):
        super(AVF_Decoder, self).__init__()
        # AVF_attention
        self.cross_att_v = att(dim, num_heads=num_heads, bias=qkv_bias, scale=qk_scale,
                               attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.cross_att_a = att(dim, num_heads=num_heads, bias=qkv_bias, scale=qk_scale,
                               attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # self attention
        self.video_self_att = att(dim, num_heads=num_heads, bias=qkv_bias, scale=qk_scale,
                                  attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.audio_self_att = att(dim, num_heads=num_heads, bias=qkv_bias, scale=qk_scale,
                                  attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # FF
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.video_Mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
        self.audio_Mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

    def AVF_Att(self, video, audio, Encoder_video, Encoder_audio):
        video_output, weight_v = self.cross_att_v((video, Encoder_audio))
        audio_output, weight_a = self.cross_att_a((audio, Encoder_video))
        return video_output, audio_output, weight_v, weight_a

    def self_att(self, video, audio):
        video_output, weight_video = self.video_self_att((video, video))
        audio_output, weight_audio = self.audio_self_att((audio, audio))
        return video_output, audio_output

    def forward(self, input):  # [b,197,768]
        video, audio, Encoder_video, Encoder_audio = input
        video_output, audio_output, weight_video, weight_audio = self.AVF_Att(video, audio,
                                                                              Encoder_video, Encoder_audio)
        video_output, audio_output = self.self_att(video_output, audio_output)

        video_output = self.video_Mlp(video_output)
        audio_output = self.audio_Mlp(audio_output)
        return video_output, audio_output, weight_video, weight_audio
