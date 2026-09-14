import torch
import torch.nn as nn
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        weight_tensor = torch.empty(out_features, in_features, device=device, dtype=dtype)#这里注意矩阵需要转置
        self.weight = nn.Parameter(weight_tensor)#需要放进参数方便更新
        sigma = (2/(in_features+out_features))**(0.5)
        #截断正态初始化
        nn.init.trunc_normal_(self.weight, mean = 0, std=sigma, a = -3*sigma, b = 3*sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x@self.weight.T#矩阵乘法y=xW.t,无偏置

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        weight_tensor = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.weight = nn.Parameter(weight_tensor)
        nn.init.trunc_normal_(self.weight, mean = 0, std = 1, a = -3, b = 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        g_tensor = torch.ones(d_model, device=device, dtype=dtype)
        self.weight = nn.Parameter(g_tensor)
        # self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        #performing RMSNorm
        rms = torch.sqrt(torch.mean(x**2, dim = -1, keepdim = True)+self.eps)
        #dim=-1表示对最后一维d_model做平方均值计算，keepdim = True表示这一维做完均值处理后不删去
        result = (x/rms)*self.weight
        result = result.to(in_dtype)
        return result


