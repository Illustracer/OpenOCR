import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Optional, Union
from openrec.modeling.common import Mlp
from openrec.modeling.encoders.svtrnet import Attention, ConvBNLayer, Block
from openrec.modeling.decoders.ctc_decoder import EncoderWithSVTR, CTCDecoder


def _fuse_modules(
    model: nn.Module,
    modules_to_fuse: Union[list[str], list[list[str]]],
    is_qat: Optional[bool],
    **kwargs: Any,
):
    if is_qat is None:
        is_qat = model.training
    method = (
        torch.ao.quantization.fuse_modules_qat
        if is_qat
        else torch.ao.quantization.fuse_modules
    )
    return method(model, modules_to_fuse, **kwargs)


class PACT(nn.Module):
    """
    PACT: Parameterized Clipping Activation for Quantization
    Paper: https://arxiv.org/abs/1805.06085
    
    将激活值裁剪到 [-alpha, alpha] 范围内，alpha 是可学习参数
    """
    def __init__(self, alpha_init=20.0, learn_alpha=True):
        super(PACT, self).__init__()
        
        # 创建可学习的 alpha 参数
        if learn_alpha:
            self.alpha = nn.Parameter(
                torch.tensor([alpha_init], dtype=torch.float32)
            )
        else:
            self.register_buffer(
                'alpha', 
                torch.tensor([alpha_init], dtype=torch.float32)
            )
        self.learn_alpha = learn_alpha

    def forward(self, x):
        """
        将 x 裁剪到 [-alpha, alpha] 范围
        等价于: x = clip(x, -alpha, alpha)
        """
        x = torch.clamp(x, min=-self.alpha, max=self.alpha)
        return x
    
    def extra_repr(self):
        return f'alpha={self.alpha.item():.4f}, learn_alpha={self.learn_alpha}'


class QuantizedAttention(Attention):
    def __init__(
        self,
        dim,
        num_heads=8,
        mixer='Global',
        HW=None,
        local_k=[7, 11],
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_pact=True,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mixer=mixer,
            HW=HW,
            local_k=local_k,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        # qkv 前后的量化/反量化
        self.quant_qkv = torch.quantization.QuantStub()
        self.dequant_qkv = torch.quantization.DeQuantStub()
        
        # proj 前后的量化/反量化
        self.quant_proj = torch.quantization.QuantStub()
        self.dequant_proj = torch.quantization.DeQuantStub()

        # PACT 预处理层（在量化之前）
        self.use_pact = use_pact
        if use_pact:
            pact_alpha_init = 20.0
            self.pact1 = PACT(alpha_init=pact_alpha_init)  # fc1 输入前
            self.pact2 = PACT(alpha_init=pact_alpha_init)  # fc2 输入前

    def forward(self, x):
        B, N, _ = x.shape

        # qkv 量化计算
        if self.use_pact:
            x = self.pact1(x)
        x = self.quant_qkv(x)      # 量化输入
        qkv = self.qkv(x)           # 量化的 Linear
        qkv = self.dequant_qkv(qkv) # 反量化输出

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.mixer == 'Local':
            attn += self.mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, self.dim)

        # proj 量化计算
        if self.use_pact:
            x = self.pact2(x)
        x = self.quant_proj(x)      # 量化输入
        x = self.proj(x)            # 量化的 Linear
        x = self.dequant_proj(x)    # 反量化输出

        x = self.proj_drop(x)
        return x

    def disable_svtr_quantization(self):
        self.attn_drop.qconfig = None
        self.proj_drop.qconfig = None
        if self.use_pact:
            self.pact1.qconfig = None
            self.pact2.qconfig = None


class QuantizedMlp(Mlp):
    # MLP 模型无需编写量化代码
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        use_pact=True,
    ):
        super().__init__(in_features, hidden_features, out_features, act_layer, drop)

        self.quant1 = torch.quantization.QuantStub()
        self.dequant1 = torch.quantization.DeQuantStub()
        self.quant2 = torch.quantization.QuantStub()
        self.dequant2 = torch.quantization.DeQuantStub()

        # PACT 预处理层（在量化之前）
        self.use_pact = use_pact
        if use_pact:
            pact_alpha_init = 20.0
            self.pact1 = PACT(alpha_init=pact_alpha_init)  # fc1 输入前
            self.pact2 = PACT(alpha_init=pact_alpha_init)  # fc2 输入前

    def forward(self, x):
        # PACT 预处理（裁剪激活值范围）
        if self.use_pact:
            x = self.pact1(x)
        x = self.quant1(x)
        x = self.fc1(x)
        x = self.dequant1(x)
        x = self.act(x)
        x = self.drop(x)

        # PACT 预处理
        if self.use_pact:
            x = self.pact2(x)
        x = self.quant2(x)
        x = self.fc2(x)
        x = self.dequant2(x)
        x = self.drop(x)
        return x

    def fuse_model(self, is_qat=None):
        """无需融合"""
        pass

    def disable_svtr_quantization(self):
        self.act.qconfig = None
        for module in self.act.modules():
            module.qconfig = None
        self.drop.qconfig = None
        if self.use_pact:
            self.pact1.qconfig = None
            self.pact2.qconfig = None


class QuantizedConvBNLayer(ConvBNLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=False,
        groups=1,
        act="swish",  # 固定为swish
        use_pact=False,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, bias, groups, act
        )
        # NOTE: 这里必须这样写, 否则有问题
        self.act = nn.Hardswish(inplace=True)

        # 量化/反量化桩
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.use_pact = use_pact
        if use_pact:
            pact_alpha_init = 20.0
            self.pact = PACT(alpha_init=pact_alpha_init)  # fc1 输入前

    def forward(self, inputs):
        if self.use_pact:
            inputs = self.pact(inputs)
        inputs = self.quant(inputs)
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.dequant(out)
        out = self.act(out)
        return out

    def fuse_model(self, is_qat=None):
        """融合Conv+BN操作"""
        _fuse_modules(self, [["conv", "norm"]], inplace=True, is_qat=is_qat)

    def disable_svtr_quantization(self):
        self.act.qconfig = None
        if self.use_pact:
            self.pact.qconfig = None


class QuantizedBlock(Block):
    def __init__(
        self,
        dim,
        num_heads,
        mixer='Global',
        local_mixer=[7, 11],
        HW=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer='nn.LayerNorm',
        eps=1e-6,
        prenorm=True,
        use_pact=True,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mixer=mixer,
            local_mixer=local_mixer,
            HW=HW,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            eps=eps,
            prenorm=prenorm,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = QuantizedMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            use_pact=use_pact,
        )

        if mixer == 'Global' or mixer == 'Local':
            self.mixer = QuantizedAttention(
                dim,
                num_heads=num_heads,
                mixer=mixer,
                HW=HW,
                local_k=local_mixer,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                use_pact=use_pact,
            )

    def forward(self, x):
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def disable_svtr_quantization(self):
        self.mlp.disable_svtr_quantization()
        self.mixer.disable_svtr_quantization()
        self.drop_path.qconfig = None
        self.norm1.qconfig = None
        self.norm2.qconfig = None


class QuantizedEncoderWithSVTR(EncoderWithSVTR):
    def __init__(
        self,
        in_channels,
        dims=64,
        depth=2,
        hidden_dims=120,
        use_guide=False,
        num_heads=8,
        qkv_bias=True,
        mlp_ratio=2.0,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path=0.0,
        kernel_size=[3, 3],
        qk_scale=None,
        use_pool=True,
        support_ppocr_v4=False,
        use_pact=False,
    ):
        # 先初始化父类以获得所有基本属性
        super().__init__(
            in_channels=in_channels,
            dims=dims,
            depth=depth,
            hidden_dims=hidden_dims,
            use_guide=use_guide,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path=drop_path,
            kernel_size=kernel_size,
            qk_scale=qk_scale,
            use_pool=use_pool,
            support_ppocr_v4=support_ppocr_v4,
        )

        # 替换ConvBNLayer为量化版本
        self.conv1 = QuantizedConvBNLayer(
            in_channels,
            in_channels // 8,
            kernel_size=kernel_size,
            padding=[kernel_size[0] // 2, kernel_size[1] // 2],
            act="swish",
            bias=False,
            use_pact=False,
        )

        self.conv2 = QuantizedConvBNLayer(
            in_channels // 8, hidden_dims, kernel_size=1, act="swish", bias=False, use_pact=False,
        )

        self.conv3 = QuantizedConvBNLayer(
            hidden_dims, in_channels, kernel_size=1, act="swish", bias=False, use_pact=False,
        )

        # conv4根据support_ppocr_v4参数设置
        if support_ppocr_v4:
            self.conv4 = QuantizedConvBNLayer(
                2 * in_channels, in_channels // 8, padding=1, act="swish", bias=False, use_pact=False,
            )
        else:
            self.conv4 = QuantizedConvBNLayer(
                2 * in_channels,
                in_channels // 8,
                kernel_size=kernel_size,
                padding=[kernel_size[0] // 2, kernel_size[1] // 2],
                act="swish",
                bias=False,
                use_pact=False,
            )

        self.conv1x1 = QuantizedConvBNLayer(
            in_channels // 8, dims, kernel_size=1, act="swish", bias=False, use_pact=False,
        )

        self.svtr_block = nn.ModuleList([
            QuantizedBlock(
                dim=hidden_dims,
                num_heads=num_heads,
                mixer='Global',
                HW=None,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer='swish',
                attn_drop=attn_drop_rate,
                drop_path=drop_path,
                norm_layer='nn.LayerNorm',
                eps=1e-05,
                prenorm=False,
                use_pact=use_pact,
            ) for i in range(depth)
        ])

    def forward(self, x):
        if self.use_pool:
            x = self.pool_h_2(x)

        # for use guide
        if self.use_guide:
            z = x.detach()
        else:
            z = x

        # for short cut
        h = z

        # reduce dim - 使用量化的ConvBN层
        z = self.conv1(z)
        z = self.conv2(z)

        # SVTR global block - 保持原有精度
        B, C, H, W = z.shape
        z = z.flatten(2).transpose(1, 2).contiguous()

        # 需要考虑 svtr_block 如何进行量化
        for blk in self.svtr_block:
            z = blk(z)
        z = self.norm(z)

        # last stage - 使用量化的ConvBN层
        z = z.reshape(-1, H, W, C).permute(0, 3, 1, 2)
        z = self.conv3(z)

        # 使用FloatFunctional进行concat操作
        z = torch.cat((h, z), dim=1)
        z = self.conv1x1(self.conv4(z))
        return z

    def fuse_model(self, is_qat=None):
        """融合"""
        self.conv1.fuse_model(is_qat)
        self.conv2.fuse_model(is_qat)
        self.conv3.fuse_model(is_qat)
        self.conv4.fuse_model(is_qat)
        self.conv1x1.fuse_model(is_qat)

    def disable_svtr_quantization(self):
        """禁用SVTR相关模块的量化"""
        # 禁用LayerNorm
        if hasattr(self, "norm"):
            self.norm.qconfig = None

        self.conv1.disable_svtr_quantization()
        self.conv2.disable_svtr_quantization()
        self.conv3.disable_svtr_quantization()
        self.conv4.disable_svtr_quantization()
        self.conv1x1.disable_svtr_quantization()

        # 禁用SVTR blocks
        for blk in self.svtr_block:
            blk.qconfig = None
            blk.disable_svtr_quantization()


class QuantizedCTCDecoder(CTCDecoder):
    """简化的CTC解码器 - 继承原始CTCDecoder"""
    # example_inputs = torch.rand(1, 120, 8, 32).to("cpu")
    def __init__(
        self,
        in_channels,
        out_channels=6625,
        mid_channels=None,
        return_feats=False,
        svtr_encoder=None,
        **kwargs,
    ):
        # 临时保存svtr_encoder配置
        if svtr_encoder is not None:
            # 暂时移除svtr_encoder配置，让父类不创建编码器
            svtr_encoder_ = None
            super().__init__(120, out_channels, mid_channels, return_feats, svtr_encoder_, **kwargs)

            # 手动创建量化版本的SVTR编码器
            svtr_encoder["in_channels"] = in_channels  # in_channels
            self.svtr_encoder = QuantizedEncoderWithSVTR(**svtr_encoder)
        else:
            super().__init__(in_channels, out_channels, mid_channels, return_feats, svtr_encoder, **kwargs)

        # 添加量化控制
        self.fc_quant = torch.quantization.QuantStub()
        self.fc_dequant = torch.quantization.DeQuantStub()

        self.use_pact = kwargs.get('use_pact', False)
        if self.use_pact:
            pact_alpha_init = 20.0
            self.pact = PACT(alpha_init=pact_alpha_init)  # fc1 输入前

    def forward(self, x):
        # SVTR编码器处理（如果存在）
        if self.svtr_encoder is not None:
            x = self.svtr_encoder(x)
            x = x.flatten(2).transpose(1, 2)

        # 输入量化
        if self.use_pact:
            x = self.pact(x)
        x = self.fc_quant(x)

        # 全连接层处理
        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)

        # 反量化
        predicts = self.fc_dequant(predicts)

        # 返回结果处理
        if self.return_feats:
            # 反量化用于返回
            x_float = self.fc_dequant(x.clone())
            result = (x_float, predicts)
        else:
            result = predicts

        # 推理时的softmax
        if not self.training:
            predicts = F.softmax(predicts, dim=2)
            result = predicts
        return result

    def fuse_model(self, is_qat=None):
        """融合模型"""
        if self.svtr_encoder is not None:
            self.svtr_encoder.fuse_model(is_qat)

    def disable_svtr_quantization(self):
        """禁用SVTR相关模块的量化"""
        self.svtr_encoder.disable_svtr_quantization()
