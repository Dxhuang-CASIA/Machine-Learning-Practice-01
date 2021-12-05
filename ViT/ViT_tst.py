from functools import partial
from collections import OrderedDict
import torch
from torch import nn

def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std = .01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode = "fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

def drop_path(input, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0], ) + (1, ) * (input.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype = input.dtype, device = input.device)
    random_tensor.floor_()
    output = input.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob = None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, input):
        return drop_path(input, self.drop_prob, self.training)

class PatchEmbed(nn.Module):
    def __init__(self, image_size = 224, patch_size = 16, in_channel = 3, embed_dim = 768, norm_layer = None):
        super(PatchEmbed, self).__init__()
        image_size = (image_size, image_size)
        patch_size = (patch_size, patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_channels = in_channel, out_channels = embed_dim,
                              kernel_size = patch_size, stride = patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, input):
        B, C, H, W = input.shape
        assert H == self.image_size[0] and W == self.image_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."

        input = self.proj(input) # [B, 3, 224, 224] => [B, 768, 14, 14]
        input = input.flatten(2) # [B, 768, 14, 14] => [B, 768, 196]
        input = input.transpose(1, 2) # [B, 768, 196] => [B, 196, 768] [B, HW, dim]
        return input

class MSA(nn.Module):
    def __init__(self,
                 dim, # 输入token的dim base对应的是768
                 num_heads = 8,
                 qkv_bias = False,
                 qk_scale = None, # sqrt(d_k)
                 attn_drop_ratio = 0.,
                 proj_drop_ratio = 0.):
        super(MSA, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.qkv = nn.Linear(dim, dim * 3, bias = qkv_bias) # WQ WK WV [dim, dim]
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim) # concat之后再进行一次投影
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, input):
        B, N, C = input.shape # [B, 196 + 1, 768]

        qkv = self.qkv(input) # [B, N, 3 * dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads) # [B, N, 3, num_heads, C / num_heads]
        qkv = qkv.permute(2, 0, 3, 1, 4) # [3, B, num_heads, N, C / num_heads]

        Q, K, V = qkv[0], qkv[1], qkv[2] # [B, num_heads, N, C / num_heads]

        # 算注意力得分
        attn = (Q @ K.transpose(-2, -1)) * self.scale # [B, num_heads, N, N]
        attn = attn.softmax(dim = -1) # 按行来
        attn = self.attn_drop(attn)

        # 加权
        # attn: [B, num_heads, N, N] V: [B, num_heads, N, C / num_heads]
        # => [B, num_heads, N, C / num_heads] => [B, N, num_heads, C / num_heads] =>[B, N, C]
        input = (attn @ V).transpose(1, 2).reshape(B, N, C)
        input = self.proj(input)
        input = self.proj_drop(input)
        return input

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features = None, out_features = None, act_layer = nn.GELU, drop = 0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, input):
        input = self.fc1(input)
        input = self.act(input)
        input = self.drop(input)
        input = self.fc2(input)
        input = self.drop(input)
        return input

class Block(nn.Module): # 编码器的一个Block 可以堆叠多个
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio = 4., # MSA后的那个MLP的隐藏层个数是dim的4倍
                 qkv_bias = False,
                 qk_scale = None,
                 drop_ratio = 0.,
                 attn_drop_ratio = 0.,
                 drop_path_ratio = 0.,
                 act_layer = nn.GELU,
                 norm_layer = nn.LayerNorm):
        super(Block, self).__init__()

        self.norm1 = norm_layer(dim)
        self.attn = MSA(dim, num_heads = num_heads, qkv_bias = qkv_bias, qk_scale = qk_scale,
                        attn_drop_ratio = attn_drop_ratio, proj_drop_ratio = drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features = dim, hidden_features = mlp_hidden_dim, act_layer = act_layer, drop = drop_ratio)

    def forward(self, input):
        input = input + self.drop_path(self.attn(self.norm1(input)))
        input = input + self.drop_path(self.mlp(self.norm2(input)))
        return input

class VisionTransformer(nn.Module):
    def __init__(self, image_size = 224, patch_size = 16, in_channel = 3, num_classes = 1000,
                 embed_dim = 768, depth = 12, num_heads = 12, mlp_ratio = 4., qkv_bias = True,
                 qk_scale = None, representation_size = None, distilled = False, # ViT用不到
                 drop_ratio = 0., attn_drop_ratio = False, drop_path_ratio = 0., embed_layer = PatchEmbed,
                 norm_layer = None, act_layer = None):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1 # ViT是1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps = 1e-6) # 传入默认参数 eps
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(image_size = image_size, patch_size = patch_size, in_channel = in_channel, embed_dim = embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p = drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, qkv_bias = qkv_bias, qk_scale = qk_scale,
                  drop_ratio = drop_ratio, attn_drop_ratio = attn_drop_ratio, drop_path_ratio = dpr[i],
                  norm_layer = norm_layer, act_layer = act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # 分类
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # 权重初始化
        nn.init.trunc_normal_(self.pos_embed, std = 0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std = 0.02)
        nn.init.trunc_normal_(self.cls_token, std = 0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, input):
        # [B, C, H, W] => [B, num_patches, embed_dim] [B, 3, 224, 224] => [B, 196, 768]
        input = self.patch_embed(input)

        cls_token = self.cls_token.expand(input.shape[0], -1, -1) # [1, 1, 768] => [B, 1, 768] 复制B次堆叠
        if self.dist_token is None: # ViT
            input = torch.cat((cls_token, input), dim = 1) # [B, 197, 768]
        else:
            input = torch.cat((cls_token, self.dist_token.expand(input.shape[0], -1, -1), input), dim = 1)

        input = self.pos_drop(input + self.pos_embed)
        input = self.blocks(input)
        input = self.norm(input)

        if self.dist_token is None:
            return self.pre_logits(input[:, 0]) # 第一个输出 第一维度batch不管
        else:
            return input[:, 0], input[:, 1]

    def forward(self, input):
        input = self.forward_features(input)
        if self.head_dist is not None: # ViT是None
            input, input_dist = self.head(input[0]), self.head_dist(input[1])
            if self.training and not torch.jit.is_scripting():
                return input, input_dist
            else:
                return (input + input_dist) / 2
        else:
            input = self.head(input)
        return input

def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits : bool = True):
    model = VisionTransformer(image_size = 224,
                              patch_size = 16,
                              embed_dim = 768,
                              depth = 12,
                              num_heads = 12,
                              representation_size = 768 if has_logits else None,
                              num_classes = num_classes)
    return model

# X = torch.rand((1, 3, 224, 224))
# model = vit_base_patch16_224_in21k()
# print(model(X))