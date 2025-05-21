import torch
import torch.nn as nn

from eval_clip import *
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from copy import deepcopy
import torch.nn.functional as F

_tokenizer = _Tokenizer()

__all__ = ['evalcoop', 'EvalCoop']

ASC = False  # atrributes specific context

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MLCPromptLearner(nn.Module):
    def __init__(self, attributes, clip_model):
        super().__init__()
        n_attrs = len(attributes)
        n_ctx_pos = 64    # 16
        n_ctx_neg = 64  # 16
        ctx_init_pos = ""    # ""
        ctx_init_neg = ""    # ""
        dtype = clip_model.dtype    # torch.float32
        ctx_dim = clip_model.ln_final.weight.shape[0]    # 512

        if ctx_init_pos and ctx_init_neg:
            # use given words to initialize context vectors
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            n_ctx_pos = len(ctx_init_pos.split(" "))    # 1
            n_ctx_neg = len(ctx_init_neg.split(" "))
            prompt_pos = eval_clip.tokenize(ctx_init_pos)    # [1, 77]
            prompt_neg = eval_clip.tokenize(ctx_init_neg)
            with torch.no_grad():
                embedding_pos = clip_model.token_embedding(prompt_pos).type(dtype)    # [1, 77, 512]
                embedding_neg = clip_model.token_embedding(prompt_neg).type(dtype)
            ctx_vectors_pos = embedding_pos[0, 1: 1 + n_ctx_pos, :]     # [16, 512]
            ctx_vectors_neg = embedding_neg[0, 1: 1 + n_ctx_neg, :]
            prompt_prefix_pos = ctx_init_pos    # ""
            prompt_prefix_neg = ctx_init_neg
            if ASC:        # attributes specific contex
                ctx_vectors_pos_ = []
                ctx_vectors_neg_ = []
                for _ in range(n_attrs):
                    ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))
                    ctx_vectors_neg_.append(deepcopy(ctx_vectors_neg))
                ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)      # [n_attrs, 16, 512]
                ctx_vectors_neg = torch.stack(ctx_vectors_neg_, dim=0)

        else:
            # Random Initialization
            if ASC:
                print("Initializing attributes-specific contexts")
                ctx_vectors_pos = torch.empty(n_attrs, n_ctx_pos, ctx_dim, dtype=dtype)       # [n_attrs, 16, 512]
                ctx_vectors_neg = torch.empty(n_attrs, n_ctx_neg, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors_pos = torch.empty(n_ctx_pos, ctx_dim, dtype=dtype)      # [16, 512]
                ctx_vectors_neg = torch.empty(n_ctx_neg, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)     # 'X X X X X X X X X X X X X X X X' 16 characters
            prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)

        print(f'Initial positive context: "{prompt_prefix_pos}"')
        print(f'Initial negative  context: "{prompt_prefix_neg}"')
        print(f"Number of positive context words (tokens): {n_ctx_pos}")
        print(f"Number of negative context words (tokens): {n_ctx_neg}")

        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized [16, 512] or [n_attrs, 16, 512]
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized

        attributes = [attribute.replace("_", " ") for attribute in attributes]
        attr_lens = [len(_tokenizer.encode(attribute)) for attribute in attributes]   # [1,2,2,2,2,3,...] length is n_attrs
        prompts_pos = [prompt_prefix_pos + " " + attribute + "." for attribute in attributes]     # ['X X X X X X X X X X X X X X X X class_name.']  length is n_attrs
        prompts_neg = [prompt_prefix_neg + " " + attribute + "." for attribute in attributes]

        tokenized_prompts_pos = []
        tokenized_prompts_neg = []
        for p_pos, p_neg in zip(prompts_pos, prompts_neg):
            tokenized_prompts_pos.append(eval_clip.tokenize(p_pos))
            tokenized_prompts_neg.append(eval_clip.tokenize(p_neg))
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)    # [n_attrs, 77]
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)   # [n_attrs, 77, 512]
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix_pos", embedding_pos[:, :1, :] )  # [n_attrs, 1, 512]
        self.register_buffer("token_suffix_pos", embedding_pos[:, 1 + n_ctx_pos:, :])   # [n_attrs, 77-16, 512]
        self.register_buffer("token_prefix_neg", embedding_neg[:, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, 1 + n_ctx_neg:, :])

        self.n_attrs = n_attrs
        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        tokenized_prompts = torch.cat([tokenized_prompts_neg, tokenized_prompts_pos], dim=0)  # torch.Tensor
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.attr_lens = attr_lens

    def forward(self):
        ctx_pos = self.ctx_pos     # [16, 512] or [n_attrs, 16, 512]
        ctx_neg = self.ctx_neg

        if ctx_pos.dim() == 2:
            ctx_pos = ctx_pos.unsqueeze(0).expand(self.n_attrs, -1, -1)   # [n_attrs, 77, 512]

        if ctx_neg.dim() == 2:
            ctx_neg = ctx_neg.unsqueeze(0).expand(self.n_attrs, -1, -1)

        prefix_pos = self.token_prefix_pos
        prefix_neg = self.token_prefix_neg
        suffix_pos = self.token_suffix_pos
        suffix_neg = self.token_suffix_neg

        prompts_pos = torch.cat(
            [
                prefix_pos,  # (n_attrs, 1, dim)
                ctx_pos,  # (n_attrs, n_ctx, dim)
                suffix_pos,  # (n_attrs, *, dim)
            ],
            dim=1,
        )

        prompts_neg = torch.cat(
            [
                prefix_neg,  # (n_attrs, 1, dim)
                ctx_neg,  # (n_attrs, n_ctx, dim)
                suffix_neg,  # (n_attrs, *, dim)
            ],
            dim=1,
        )

        prompts = torch.cat([prompts_neg, prompts_pos], dim=0)

        tokenized_prompts = self.tokenized_prompts

        return prompts, tokenized_prompts


class EvalCoop(nn.Module):
    def __init__(self, attributes, clip_model):
        super().__init__()
        self.prompt_learner = MLCPromptLearner(attributes, clip_model)

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = 100.0
        self.dtype = clip_model.dtype

    def forward(self, image):
        # get image and text features
        image_features, attn_weights = self.image_encoder(image.type(self.dtype))
        prompts, tokenized_prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)

        # normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)

        # Class-Specific Region Feature Aggregation
        output = 20 * F.conv1d(image_features_norm, text_features[:, :, None])  # input [100, 1024, 50] weight [34, 1024, 1] out [100, 34, 50]
        b, c, _ = output.shape
        output_half = output[:,  c // 2:]   # [100, 17, 50]
        w_half = F.softmax(output_half, dim=-1)    # [100, 17, 50]
        w = torch.cat([w_half, w_half], dim=1)  # [100, 34, 50]
        output = 5 * (output * w).sum(-1)   # [100, 34]

        b, c = output.shape

        # convert the shape of logits to [b, 2, num_attrs]
        logits = output.resize(b, 2, c//2)  # [100, 2, 17]
        Softmax = torch.nn.Softmax(dim=1)
        prob = Softmax(logits)[:, 1]

        return prob

    def backbone_params(self):
        params = []
        for name, param in self.named_parameters():
            if "image_encoder" in name and "prompt_learner" not in name and 'attnpool' not in name:
                params.append(param)
        return params

    def attn_params(self):
        params = []
        for name, param in self.named_parameters():
            if 'attnpool' in name and 'image_encoder' in name:
                params.append(param)
                print(name)
        return params

    def prompt_params(self):
        params = []
        for name, param in self.named_parameters():
            if "prompt_learner" in name:
                params.append(param)
        return params


def evalcoop(attributes, **kwargs):
    print(f"Loading CLIP")
    clip_model, _ = eval_clip.load("RN101", device='cpu')

    clip_model.float()

    print("Building evalcoop")
    model = EvalCoop(attributes, clip_model)

    print('Freeze the backbone weights')
    backbone_params = model.backbone_params()
    for param in backbone_params:
        param.requires_grad_(False)

    print('Freeze the attn weights')
    attn_params = model.attn_params()
    for param in attn_params:
        param.requires_grad_(False)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    # Note that multi-gpu training could be slow because CLIP's size is
    # big, which slows down the copy operation in DataParallel
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        model = nn.DataParallel(model)
    return model
