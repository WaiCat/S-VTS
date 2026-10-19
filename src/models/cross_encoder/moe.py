import torch
import torch.nn as nn


class ExpertMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = max(int(config.cross_moe_num_experts), 1)
        self.top_k = min(max(int(config.cross_moe_top_k), 1), self.num_experts)
        self.input_size = int(config.cross_moe_input_size)
        self.output_size = int(config.cross_moe_output_size)
        self.hidden_size = int(config.cross_moe_hidden_size)
        self.dropout_prob = float(getattr(config, "hidden_dropout_prob", 0.1))
        self.balance_loss_weight = float(getattr(config, "cross_moe_lw", 1.0))
        self.use_residual = bool(getattr(config, "cross_moe_residual", True))

        self.gate = nn.Linear(self.input_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                ExpertMLP(self.input_size, self.hidden_size, self.output_size, self.dropout_prob)
                for _ in range(self.num_experts)
            ]
        )
        self.last_stats = {}

    @staticmethod
    def cv_squared(x):
        eps = 1e-10
        if x.numel() <= 1:
            return x.new_tensor(0.0)
        return x.float().var(unbiased=False) / (x.float().mean().pow(2) + eps)

    def _store_stats(self, gate_probs, topk_indices, importance, load, importance_loss, load_loss, balance_loss):
        token_count = max(gate_probs.shape[0], 1)
        selection_counts = torch.zeros_like(load)
        selection_counts.scatter_add_(
            0,
            topk_indices.reshape(-1),
            torch.ones(topk_indices.numel(), device=topk_indices.device, dtype=load.dtype),
        )
        self.last_stats = {
            "importance": importance.detach(),
            "load": load.detach(),
            "selection_ratio": (selection_counts / float(token_count)).detach(),
            "importance_loss": importance_loss.detach(),
            "load_loss": load_loss.detach(),
            "balance_loss": balance_loss.detach(),
        }

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError(f"MoE expects 2D input, got shape {tuple(x.shape)}")

        gate_logits = self.gate(x)
        topk_logits, topk_indices = torch.topk(gate_logits, k=self.top_k, dim=-1)
        topk_weights = torch.softmax(topk_logits, dim=-1)

        gate_probs = torch.zeros_like(gate_logits)
        gate_probs.scatter_(1, topk_indices, topk_weights)

        output = torch.zeros(x.shape[0], self.output_size, device=x.device, dtype=x.dtype)
        for expert_index, expert in enumerate(self.experts):
            expert_token_mask = topk_indices == expert_index
            if not expert_token_mask.any():
                continue
            token_indices, topk_slot_indices = expert_token_mask.nonzero(as_tuple=True)
            expert_inputs = x[token_indices]
            expert_outputs = expert(expert_inputs)
            expert_weights = topk_weights[token_indices, topk_slot_indices].unsqueeze(-1).to(expert_outputs.dtype)
            output.index_add_(0, token_indices, expert_outputs * expert_weights)

        if self.use_residual and self.input_size == self.output_size:
            output = output + x

        importance = gate_probs.sum(dim=0)
        load = (gate_probs > 0).sum(dim=0).to(importance.dtype)
        importance_loss = self.cv_squared(importance)
        load_loss = self.cv_squared(load)
        balance_loss = self.balance_loss_weight * (importance_loss + load_loss)

        self._store_stats(gate_probs, topk_indices, importance, load, importance_loss, load_loss, balance_loss)
        return output, balance_loss
