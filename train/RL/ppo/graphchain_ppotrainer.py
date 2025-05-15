from trl import PPOTrainer
from typing import List, Optional
import math
import time
import torch
import numpy as np
import warnings
from trl.core import (WANDB_PADDING,PPODecorators,convert_to_scalar,logprobs_from_logits,stack_dicts,masked_mean,stats_to_np)
from trl.models import PreTrainedModelWrapper

class GraphChainPPOTrainer(PPOTrainer):
    def _step_safety_checker(self,batch_size: int,queries: List[torch.LongTensor],responses: List[torch.LongTensor],scores: List[torch.FloatTensor],frag_masks: Optional[List[torch.LongTensor]] = None,masks: Optional[List[torch.LongTensor]] = None):
        queries = [tensor.to(self.current_device) for tensor in queries]
        responses = [tensor.to(self.current_device) for tensor in responses]
        scores = [tensor.to(self.current_device) for tensor in scores]  # TODO
        frag_masks = [tensor.to(self.current_device) for tensor in frag_masks] if frag_masks is not None else None
        masks = [tensor.to(self.current_device) for tensor in masks] if masks is not None else None
        return queries, responses, scores, frag_masks, masks
    def step(self,queries: List[torch.LongTensor],responses: List[torch.LongTensor],scores: List[torch.FloatTensor],frag_masks: Optional[List[torch.LongTensor]] = None,response_masks: Optional[List[torch.LongTensor]] = None):
        bs = self.config.batch_size
        queries, responses, scores, frag_masks, response_masks = self._step_safety_checker(bs, queries, responses, scores, frag_masks, response_masks)
        timing = dict()
        t0 = time.time()
        t = time.time()
        model_inputs = self.prepare_model_inputs(queries, responses)
        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"
            model_inputs["input_ids"] = self.accelerator.pad_across_processes(model_inputs["input_ids"],dim=1,pad_index=self.tokenizer.pad_token_id,pad_first=pad_first)
            model_inputs["attention_mask"] = self.accelerator.pad_across_processes(model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first)
            if self.is_encoder_decoder:
                model_inputs["decoder_input_ids"] = self.accelerator.pad_across_processes(model_inputs["decoder_input_ids"],dim=1,pad_index=self.tokenizer.pad_token_id,pad_first=pad_first)
                model_inputs["decoder_attention_mask"] = self.accelerator.pad_across_processes(model_inputs["decoder_attention_mask"],dim=1,pad_index=0,pad_first=pad_first)
        model_inputs_names = list(model_inputs.keys())
        full_kl_penalty = self.config.kl_penalty == "full"
        with torch.no_grad():
            all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(self.model,queries,responses,model_inputs,response_masks=response_masks,return_logits=full_kl_penalty,frag_masks=frag_masks)
            with self.optional_peft_ctx():
                ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(self.model if self.is_peft_model else self.ref_model,queries,responses,model_inputs,return_logits=full_kl_penalty,frag_masks=frag_masks)
        timing["time/ppo/forward_pass"] = time.time() - t
        with torch.no_grad():
            t = time.time()
            if full_kl_penalty:
                active_full_logprobs = logprobs_from_logits(logits_or_none, None, gather=False)
                ref_full_logprobs = logprobs_from_logits(ref_logits_or_none, None, gather=False)
                rewards, non_score_reward, kls = self.compute_rewards(scores, active_full_logprobs, ref_full_logprobs, masks)
            else:
                rewards, non_score_reward, kls = self.compute_rewards(scores, all_logprobs, ref_logprobs, masks)
            timing["time/ppo/compute_rewards"] = time.time() - t
            t = time.time()
            values, advantages, returns = self.compute_advantages(values, rewards, masks)
            timing["time/ppo/compute_advantages"] = time.time() - t
        batch_dict = {"queries": queries,"responses": responses,"logprobs": all_logprobs.to(torch.float32),"values": values.to(torch.float32),"masks": masks,"advantages": advantages,"returns": returns,"frag_masks": frag_masks}
        batch_dict.update(model_inputs)
        t = time.time()
        all_stats = []
        early_stop = False
        for _ in range(self.config.ppo_epochs):
            if early_stop:
                break
            b_inds = np.random.permutation(bs)
            for backward_batch_start in range(0, bs, self.config.backward_batch_size):
                backward_batch_end = backward_batch_start + self.config.backward_batch_size
                backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]
                for mini_batch_start in range(0, self.config.backward_batch_size, self.config.mini_batch_size):
                    mini_batch_end = mini_batch_start + self.config.mini_batch_size
                    mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]
                    mini_batch_dict = {
                        "logprobs": batch_dict["logprobs"][mini_batch_inds],
                        "values": batch_dict["values"][mini_batch_inds],
                        "frag_masks": [batch_dict["frag_masks"][i] for i in mini_batch_inds],
                        "masks": batch_dict["masks"][mini_batch_inds],
                        "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                        "responses": [batch_dict["responses"][i] for i in mini_batch_inds],
                        "advantages": batch_dict["advantages"][mini_batch_inds],
                        "returns": batch_dict["returns"][mini_batch_inds]
                    }
                    for k in model_inputs_names:
                        mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                    with self.accelerator.accumulate(self.model):
                        model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}

                        logprobs, logits, vpreds, _ = self.batched_forward_pass(
                            self.model,
                            mini_batch_dict["queries"],
                            mini_batch_dict["responses"],
                            model_inputs,
                            return_logits=True,
                            frag_masks=mini_batch_dict["frag_masks"],
                        )
                        train_stats = self.train_minibatch(
                            mini_batch_dict["logprobs"],
                            mini_batch_dict["values"],
                            logprobs,
                            logits,
                            vpreds,
                            mini_batch_dict["masks"],
                            mini_batch_dict["advantages"],
                            mini_batch_dict["returns"]
                        )
                        all_stats.append(train_stats)
            if self.config.early_stopping:
                policykl = train_stats["policy/policykl"]
                early_stop = self._early_stop(policykl)
                if early_stop:
                    break
        timing["time/ppo/optimize_step"] = time.time() - t
        t = time.time()
        train_stats = stack_dicts(all_stats)
        train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
        train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)
        train_stats["policy/advantages_batch"] = batch_dict["advantages"].mean(axis=-1).unsqueeze(0)
        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=masks,
            queries=queries,
            responses=responses,
            kls=kls,
        )
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]
        self.kl_ctl.update(
            stats["objective/kl"],
            self.config.batch_size * self.accelerator.num_processes,
        )
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)
        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return stats
    
    @PPODecorators.empty_device_cache()
    def batched_forward_pass(self,model: PreTrainedModelWrapper,queries: torch.Tensor,responses: torch.Tensor,model_inputs: dict,return_logits: bool = False,response_masks: Optional[torch.Tensor] = None,frag_masks: Optional[torch.Tensor] = None):
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []
        model.eval()
        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            frag_masks_batch = frag_masks[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
            logits, _, values = model(**input_kwargs)
            if self.is_encoder_decoder:
                input_ids = input_kwargs["decoder_input_ids"]
                attention_mask = input_kwargs["decoder_attention_mask"]
            else:
                input_ids = input_kwargs["input_ids"]
                attention_mask = input_kwargs["attention_mask"]
            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            origin_masks = torch.zeros_like(attention_mask)
            frag_masks_pad = torch.zeros_like(attention_mask) 
            for idx, f_m in enumerate(frag_masks_batch):
                frag_masks_pad[idx, :f_m.shape[0]] = f_m
            masks[:, :-1] = frag_masks_pad[:, 1:]
            origin_masks[:, :-1] = attention_mask[:, 1:]
            for j in range(len(query_batch)):
                if self.is_encoder_decoder:
                    start = 1
                    end = attention_mask[j, :].sum() - 1
                else:
                    start = len(query_batch[j]) - 1
                    if attention_mask[j, 0] == 0:
                        start += attention_mask[j, :].nonzero()[0]
                    end = start + len(response_batch[j])
                    if response_masks is not None:
                        response_masks_batch[j] = torch.cat((torch.zeros_like(query_batch[j]), response_masks_batch[j]))[1:]
                origin_masks[j, :start] = 0
                origin_masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]
            if return_logits:
                all_logits.append(logits)
            else:
                del logits
            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (torch.cat(all_logprobs),torch.cat(all_logits)[:, :-1] if return_logits else None,torch.cat(all_values)[:, :-1],torch.cat(all_masks)[:, :-1])

    def compute_rewards(self,scores: torch.FloatTensor,logprobs: torch.FloatTensor,ref_logprobs: torch.FloatTensor,masks: torch.LongTensor):
        rewards, non_score_rewards, kls = [], [], []
        for score, logprob, ref_logprob, mask in zip(scores, logprobs, ref_logprobs, masks):
            kl = self._kl_penalty(logprob, ref_logprob)
            kls.append(kl)
            non_score_reward = -self.kl_ctl.value * kl
            non_score_rewards.append(non_score_reward)

            reward = non_score_reward.clone()
            last_mask = mask.clone()
            for score_idx in reversed(range(len(score))):
                if len(last_mask.nonzero()) == 0:
                    break
                last_1_index = last_mask.nonzero()[-1]
                reward[last_1_index] += score[score_idx] * (len(score)-score_idx-1)
                last_mask = last_mask[:last_1_index+1]
                if len((last_mask==0).nonzero()) == 0:
                    break
                last_0_index = (last_mask==0).nonzero()[-1]
                last_mask = last_mask[:last_0_index+1]
            rewards.append(reward)
        return torch.stack(rewards), torch.stack(non_score_rewards), torch.stack(kls)

    def record_step_stats(self, kl_coef: float, **data):
        mask = data.pop("masks")

        kls = data.pop("kls")
        kl_list = ((kls) * mask).sum(axis=-1)
        mean_kl = kl_list.mean()
        mean_entropy = (-data["logprobs"] * mask).sum(axis=-1).mean()

        mean_non_score_reward = masked_mean(
            data["non_score_reward"], mask
        )
        final_scores = torch.tensor([score[-1] for score in data["scores"]])
        mean_scores = final_scores.mean()
        std_scores = final_scores.std()
        stats = {
            "objective/kl": mean_kl,
            "objective/kl_dist": kl_list,
            "objective/logprobs": data["logprobs"],
            "objective/ref_logprobs": data["ref_logprobs"],
            "objective/kl_coef": kl_coef,
            "objective/entropy": mean_entropy,
            "ppo/mean_non_score_reward": mean_non_score_reward,
            "ppo/mean_scores": mean_scores,
            "ppo/std_scores": std_scores,
        }
        query_lens = torch.tensor([len(query) for query in data["queries"]], dtype=torch.float)
        response_lens = torch.tensor([len(response) for response in data["responses"]], dtype=torch.float)
        stats["tokens/queries_len_mean"] = torch.mean(query_lens).cpu().numpy().item()
        stats["tokens/queries_len_std"] = torch.std(query_lens).cpu().numpy().item()
        stats["tokens/queries_dist"] = query_lens.cpu().numpy()
        stats["tokens/responses_len_mean"] = torch.mean(response_lens).cpu().numpy().item()
        stats["tokens/responses_len_std"] = torch.std(response_lens).cpu().numpy().item()
        stats["tokens/responses_dist"] = response_lens.cpu().numpy()
        for k, v in data["train_stats"].items():
            stats[f"ppo/{k}"] = torch.mean(v, axis=0)
        stats["ppo/val/var_explained"] = 1 - stats["ppo/val/error"] / stats["ppo/returns/var"]
        return stats