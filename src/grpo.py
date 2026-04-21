"""
The actual GRPO
"""
from dataclasses import dataclass, field, fields
import torch
import torch.nn.functional as F
from typing import Any, Iterator, Optional,List
import torch.optim as optim
from utils import compute_entropy_from_logits
from torch.nn.utils import clip_grad_norm_


@dataclass
class Experience:
    sequence_ids: torch.Tensor
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.Tensor]
    action_mask: torch.Tensor
    start_ids: int
    logits_to_keep: int
    gen_log_probs: torch.Tensor
    ref_log_probs: torch.Tensor


    def to(self, device: torch.device):
        members = {}
        for field in fields(self):
            v = getattr(self, field.name)
            if isinstance(v, torch.Tensor):
                v = v.to(device=device)
            members[field.name] = v
        return Experience(**members)

@dataclass
class GRPOConfig():
    num_generations: int | None = field(
        default = 12
    )
    beta: float = 0.001
    epsilon: float = 0.2
    epsilon_low: float = None
    epsilon_high: float = None
    micro_batch_size: int = 2
    steps_per_generation: int = 1
    clip_gradient: float = 1.0
    def __post_init__(self):
        if self.epsilon_low == None:
            self.epsilon_low = self.epsilon
        if self.epsilon_high == None:
            self.epsilon_high = self.epsilon_low
        return

def per_token_log_probs(logits,targets,is_logits_log = False, mem_eff = True):
    if mem_eff and logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1) # Shape (B, L)
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits]) # Shape (B, L)
        token_log_probs = selected_logits - logsumexp_values
    else:
        if not is_logits_log:
            logits = F.log_softmax(logits, dim=-1)
        token_log_probs = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return token_log_probs
    

@torch.no_grad()
def advantage_compute(rewards, std_scale = True):
    advantages = (rewards - rewards.mean())
    if rewards.shape[1] > 1 and std_scale:
        advantages /= (rewards.std() + 1e-8)
    return advantages

# computes the log probs
def sequences_log_probs(model, sequence_ids, attention_mask, start_seq=None, temperature = 1.0, batch_size = None, compute_entropy = False):

    if batch_size == None:
        batch_size = sequence_ids.shape[0]

    out = []
    entropy_out = []
    for start in range(0,sequence_ids.shape[0], batch_size):
        end = min(sequence_ids.shape[0],start + batch_size)
        _loc_sequence_ids = sequence_ids[start: end]
        _loc_attention_mask = attention_mask[start: end]
        logits = model(input_ids=_loc_sequence_ids, attention_mask=_loc_attention_mask,use_cache=False).logits
        loss_mask = _loc_attention_mask[:, start_seq:].to(dtype=logits.dtype).contiguous()
        # Remove last one (hallucinated)
        logits = logits[:, :-1, :]
        logits = logits[:, (start_seq-1):, :]
        targets = _loc_sequence_ids[:,start_seq:]
        token_log_probs = per_token_log_probs(logits,targets)
        # take the attention mask from completion start onwards
        loss_mask = _loc_attention_mask[:, start_seq:].to(dtype=logits.dtype).contiguous()

        out.append(token_log_probs * loss_mask + (1.0 - loss_mask) * torch.finfo(token_log_probs.dtype).min)
        
        if compute_entropy:
            with torch.no_grad():
                entropy_out.append(compute_entropy_from_logits(logits,256))
    if compute_entropy:
        return torch.cat(out,dim=0), torch.cat(entropy_out,dim=0)
    else:
        return torch.cat(out,dim=0), None

def grpo_loss(log_probs, advantages, action_mask, grpo_config: GRPOConfig, gen_per_token_logps = None, ref_log_probs = None, method = "nis"):
        """Compute the GRPO loss.
        """
        if "vis" in method:
            print("VIS!")
            do_ref = True
            if gen_per_token_logps == None:
                gen_per_token_logps = log_probs.detach()
                do_ref = False

            coef_1 = torch.exp(log_probs - gen_per_token_logps.detach())
            if do_ref:
                coef_2 = torch.clamp(coef_1, 1 - grpo_config.epsilon_low, 1 + grpo_config.epsilon_high)
                per_token_loss = -torch.min(coef_1 * advantages, coef_2 * advantages)
            else:
                per_token_loss = -coef_1 * advantages
            if ref_log_probs != None:
                per_token_kl = (
                    torch.exp(ref_log_probs - log_probs)
                    - (ref_log_probs - log_probs)
                    - 1
                )

                per_token_loss += grpo_config.beta * per_token_kl
        else:
            
            coef_1 = torch.exp(log_probs - log_probs.detach())
            per_token_loss = -coef_1 * advantages
            
            if "btis" in method:
                print("BTIS")
                r = torch.exp(log_probs.detach() - gen_per_token_logps.detach())
                r = torch.clamp(r, max=2.0)
                per_token_loss *= r
                c = torch.clamp(r - 2.0, min=0.0)
                per_token_loss = per_token_loss - c * advantages.detach()
            elif "tis" in method:
                print("TIS")
                r = torch.exp(log_probs.detach() - gen_per_token_logps.detach())
                r = torch.clamp(r, max=2.0)
                per_token_loss *= r
                
                per_token_loss = per_token_loss
            else:
                print("NIS")
            if ref_log_probs != None:
                per_token_kl = (
                    torch.exp(ref_log_probs - log_probs)
                    - (ref_log_probs - log_probs)
                    - 1
                )

                per_token_loss += grpo_config.beta * per_token_kl
                


        loss = (per_token_loss * action_mask).sum(dim=-1) / action_mask.sum(dim=-1)
        return loss.mean()


def grpo_train_loop(model, optimizer, replay_buffer, grpo_config: GRPOConfig, ref_model = None, compute_entropy = True, method = True, step = 0, **loss_kwargs):
    model.train()
    device = model.device
    mb_size = grpo_config.micro_batch_size
    grad_accum_steps = (len(replay_buffer) * replay_buffer[0].sequence_ids.shape[0] // mb_size)
    update_every = grad_accum_steps // grpo_config.steps_per_generation
    loss_hist = []
    tmp_loss_hist = []
    kl_hist = []
    tmp_kl_hist = []
    entropy_hist = []
    tmp_entropy_hist = []
    _steps = 0
    optimizer.zero_grad()

    for exp in replay_buffer:
        exp: Experience
        if torch.count_nonzero(exp.advantages).item() == 0:
            continue
        print("processing experience",exp.sequence_ids.shape)
        batches = exp.sequence_ids.shape[0] // mb_size
        exp = exp.to(device)
        for mb in range(batches):
            _steps += 1
            end = (mb+1) * mb_size
            rng = (mb * mb_size, min(end,exp.sequence_ids.shape[0]) )
            
            # Compute log probs
            log_probs, entropy = sequences_log_probs(
                        model, sequence_ids=exp.sequence_ids[rng[0]:rng[1],:], attention_mask=exp.attention_mask[rng[0]:rng[1],:],
                        start_seq=exp.logits_to_keep, compute_entropy = True
            )
            # Use ref log probs to compute kl-divergence:
            drop = []
            gen_log_probs = exp.gen_log_probs[rng[0]:rng[1],:]
            with torch.no_grad():
                per_token_kl = (
                    torch.exp(gen_log_probs - log_probs)
                    - (gen_log_probs - log_probs)
                    - 1
                )
                tmp_pkl = per_token_kl.mean(-1)
                print(tmp_pkl.shape)
                tmp_pkl = tmp_pkl.tolist()
            for idx,adv in enumerate(exp.advantages[rng[0]:rng[1]]):
                adv = adv.item()
                if adv <= 0 and tmp_pkl[idx] > 200:
                    # print("need to drop",idx)
                    drop.append(idx)
            foreign = "f-" in method and len(drop) > 0
            tmp_kl_hist.append(per_token_kl.mean().item())
            tmp_entropy_hist.append(entropy.mean().item())
            del entropy
            del per_token_kl
            if foreign:
                print("filtering!")
                if len(drop) == (rng[1] - rng[0]):
                    del log_probs
                    torch.cuda.empty_cache()
                    continue

            action_mask = exp.action_mask[rng[0]:rng[1],:]
            advantages = exp.advantages[rng[0]:rng[1]]
            ref_log_probs = None
                
            start_ids = exp.start_ids
            if foreign:
                
                for idx,i in enumerate(drop):
                    
                    log_probs = torch.cat([log_probs[:(i-idx),:],log_probs[(1+i-idx):,:]])
                    gen_log_probs = torch.cat([gen_log_probs[:(i-idx),:],gen_log_probs[(1+i-idx):,:]])
                    advantages = torch.cat([advantages[:(i-idx)],advantages[(1+i-idx):]])
                    action_mask = torch.cat([action_mask[:(i-idx)],action_mask[(1+i-idx):]])
                    # ref_log_probs = torch.cat([ref_log_probs[:(i-idx)],ref_log_probs[(1+i-idx):]])
            
            
            loss = grpo_loss(log_probs=log_probs, advantages=advantages, action_mask=action_mask,
                            grpo_config=grpo_config, ref_log_probs=ref_log_probs, gen_per_token_logps=gen_log_probs, method = method)

            if not loss.isfinite():
                print("INFINITE LOSS")
                continue

            print(f"loss={loss: .4f}")
            loss = loss / (update_every)
            tmp_loss_hist.append(loss.item())
            loss.backward()
            
        del exp
    if True:
        print("update")
        if grpo_config.clip_gradient != None:
            clip_grad_norm_(model.parameters(), max_norm=grpo_config.clip_gradient)
        loss_hist.append(sum(tmp_loss_hist)/len(tmp_loss_hist))
        kl_hist.append(sum(tmp_kl_hist)/len(tmp_kl_hist))
        if compute_entropy:
            entropy_hist.append(sum(tmp_entropy_hist)/len(tmp_entropy_hist))
                
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.empty_cache()
    return loss_hist, kl_hist, entropy_hist