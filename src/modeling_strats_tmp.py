from argparse import Namespace
from models import TimeSeriesModel
import torch.nn as nn
from transformers import BertModel, BertConfig
import torch
import torch.nn.functional as F
import numpy as np



class CVE(nn.Module):
    def __init__(self, args):
        super().__init__()
        int_dim = int(np.sqrt(args.hid_dim))
        self.W1 = nn.Parameter(torch.empty(1,int_dim), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros(int_dim), requires_grad=True)
        self.W2 = nn.Parameter(torch.empty(int_dim, args.hid_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        self.activation = torch.tanh

    def forward(self, x):
        # x: bsz, max_len
        x = torch.unsqueeze(x, -1)
        x = torch.matmul(x,self.W1) + self.b1[None,None,:] # bsz,max_len,int_dim
        x = self.activation(x)
        x = torch.matmul(x,self.W2) #bsz,max_len,hid_dim
        return x
        

class FusionAtt(nn.Module):
    def __init__(self, args):
        super().__init__()
        int_dim = args.hid_dim
        self.W = nn.Parameter(torch.empty(args.hid_dim,int_dim), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(int_dim), requires_grad=True)
        self.u = nn.Parameter(torch.empty(int_dim,1), requires_grad=True)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.u)
        self.activation = torch.tanh

    def forward(self, x, mask):
        # x: bsz, max_len, hid_dim
        att = torch.matmul(x,self.W) + self.b[None,None,:] # bsz,max_len,int_dim
        att = self.activation(att)
        att = torch.matmul(att,self.u)[:,:,0] # bsz,max_len
        att = att + (1-mask)*torch.finfo(att.dtype).min  
        att = torch.softmax(att,dim=-1) # bsz,max_len
        return att 
    

class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.N = args.num_layers
        self.d = args.hid_dim
        self.dff = self.d*2
        self.attention_dropout = args.attention_dropout
        self.dropout = args.dropout
        self.h = args.num_heads
        self.dk = self.d//self.h
        self.all_head_size = self.dk*self.h

        self.Wq = nn.Parameter(self.init_proj((self.N, self.h, self.d, self.dk)), requires_grad=True)
        self.Wk = nn.Parameter(self.init_proj((self.N, self.h, self.d, self.dk)), requires_grad=True)
        self.Wv = nn.Parameter(self.init_proj((self.N, self.h, self.d, self.dk)), requires_grad=True)
        self.Wo = nn.Parameter(self.init_proj((self.N, self.all_head_size, self.d)), requires_grad=True)
        self.W1 = nn.Parameter(self.init_proj((self.N, self.d, self.dff)), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros((self.N, 1, 1, self.dff)), requires_grad=True)
        self.W2 = nn.Parameter(self.init_proj((self.N, self.dff, self.d)), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros((self.N, 1, 1, self.d)), requires_grad=True)
        # self.layer_norm1 = nn.ModuleList([nn.LayerNorm(self.d) for i in range(self.N)])
        # self.layer_norm2 = nn.ModuleList([nn.LayerNorm(self.d) for i in range(self.N)])

    def init_proj(self, shape, gain=1):
        x = torch.rand(shape)
        fan_in_out = shape[-1]+shape[-2]
        scale = gain*np.sqrt(6/fan_in_out)
        x = x*2*scale - scale
        return x
    
    def forward(self, x, mask):
        # x: bsz, max_len, d
        # mask: bsz, max_len
        bsz, max_len, _ = x.size()
        mask = mask[:,:,None]*mask[:,None,:]
        mask = (1-mask)[:,None,:,:]*torch.finfo(x.dtype).min
        layer_mask = mask
        for i in range(self.N):
            # MHA
            q = torch.einsum('bld,hde->bhle', x, self.Wq[i]) # bsz,h,max_len,dk
            k = torch.einsum('bld,hde->bhle', x, self.Wk[i]) # bsz,h,max_len,dk
            v = torch.einsum('bld,hde->bhle', x, self.Wv[i]) # bsz,h,max_len,dk
            # x_for_qkv = x[:,None,:,:,None]
            # q = (x_for_qkv*self.Wq[i][None,:,None,:,:]).sum(dim=-2) # bsz,h,max_len,dk
            # k = (x_for_qkv*self.Wk[i][None,:,None,:,:]).sum(dim=-2) # bsz,h,max_len,dk
            # v = (x_for_qkv*self.Wv[i][None,:,None,:,:]).sum(dim=-2) # bsz,h,max_len,dk
            A = torch.einsum('bhle,bhke->bhlk', q, k) # bsz,h,max_len,max_len
            # A = (q[:,:,:,None,:]*k[:,:,None,:,:]).sum(dim=-1) # bsz,h,max_len,max_len
            if self.training:
                dropout_mask = (torch.rand_like(A)<self.attention_dropout
                                ).float()*torch.finfo(x.dtype).min
                layer_mask = mask+dropout_mask
            A = A+layer_mask
            A = torch.softmax(A, dim=-1)
            v = torch.einsum('bhkl,bhle->bkhe',A,v) # bsz,max_len,h,dk
            # v = (A[:,:,:,:,None]*v[:,:,None,:,:]).sum(dim=-2).transpose(1,2) # bsz,max_len,h,dk
            all_head_op = v.reshape((bsz, max_len, -1))
            all_head_op = torch.matmul(all_head_op, self.Wo[i])
            all_head_op = F.dropout(all_head_op, self.dropout, self.training)
            # Add+layernorm
            # x = self.layer_norm1[i](all_head_op+x) # bsz,max_len,d
            x = (all_head_op+x)/2
            # FFN
            ffn_op = torch.matmul(x,self.W1[i])+self.b1[i]
            ffn_op = F.gelu(ffn_op)
            ffn_op = torch.matmul(ffn_op,self.W2[i])+self.b2[i]
            ffn_op = F.dropout(ffn_op, self.dropout, self.training)
            # Add+layernorm
            # x = self.layer_norm2[i](ffn_op+x)
            x = (ffn_op+x)/2
        return x




class Strats(TimeSeriesModel):
    def __init__(self, args: Namespace):
        super().__init__(args)
        self.cve_time = CVE(args)
        self.cve_value = CVE(args)
        self.variable_emb = nn.Embedding(args.V, args.hid_dim)
        self.transformer = Transformer(args)
        self.fusion_att = FusionAtt(args)
        self.dropout = args.dropout
        self.V = args.V
        

    def forward(self, values, times, varis, obs_mask, demo,
                labels=None, forecast_values=None, forecast_mask=None, return_interpret=False):
        bsz, max_obs = values.size()
        device = values.device
        if self.training:
            with torch.no_grad():
                var_mask = (torch.rand((bsz,self.V), device=device)<=self.dropout).int()
                for v in range(self.V):
                    mask_pos = (varis==v).int()*var_mask[:,v:v+1]
                    obs_mask = obs_mask*(1-mask_pos)

        # demographics embedding
        demo_emb = self.demo_emb(demo) if self.args.model_type=='strats' \
                    else demo
        # initial triplet embedding
        time_emb = self.cve_time(times)
        value_emb = self.cve_value(values)
        # value_emb = 0
        # for i in range(self.args.V):
        #     value_emb = value_emb + self.cve_value[i](values) * (varis==i)
        vari_emb = self.variable_emb(varis)
        triplet_emb = time_emb+value_emb+vari_emb
        triplet_emb = F.dropout(triplet_emb, self.dropout, self.training)
        # contextual triplet emb
        contextual_emb = self.transformer(triplet_emb, obs_mask) 
        # fusion attention
        attention_weights = self.fusion_att(contextual_emb, obs_mask)[:,:,None]
        if self.args.model_type=='istrats':
            ts_emb = (triplet_emb*attention_weights).sum(dim=1)
        else:
            ts_emb = (contextual_emb*attention_weights).sum(dim=1)
        # concat demo and ts_emb
        ts_demo_emb = torch.cat((ts_emb, demo_emb), dim=-1)
        # prediction/loss
        if self.pretrain:
            return self.forecast_final(ts_demo_emb, forecast_values, forecast_mask)
        logits = self.binary_head(self.forecast_head(ts_demo_emb))[:,0] \
                    if self.finetune else self.binary_head(ts_demo_emb)[:,0]
        if not return_interpret:
            return self.binary_cls_final(logits, labels)
        
        ## interpretable contributions for I-Strats

        if self.args.model_type=='istrats' and return_interpret:
            with torch.no_grad():
                # sigmoid prediction
                probs = torch.sigmoid(logits)
                # get final linear weights w_o (shape: D + d,) and bias
                # binary_head is Linear(in_features=ts_demo_emb_size, out_features=1)
                w_o = self.binary_head.weight.squeeze(0)  # (D + d,)
                b_o = self.binary_head.bias.squeeze(0)

                D = self.args.D  # size of demo vector
                # split weights into demo and triplet parts
                w_demo = w_o[:D]          # (D,)
                w_trip = w_o[D:]          # (d,)

                # contribution of demographics per sample: B
                demo_contrib = (demo_emb * w_demo).sum(dim=1)

                # per-observation contribution: B x N
                # for istrats, time-series embedding is sum_i alpha_i * e_i
                # contribution_i = alpha_i * <w_trip, triplet_emb_i>
                inner = (triplet_emb * w_trip[None, None, :]).sum(dim=2)  # (B, N)
                att = attention_weights.squeeze(-1) 

                obs_contrib = att * inner   

                ts_contrib = obs_contrib.sum(dim=1)
            
            # return dict in eval for interpretability
            if labels is not None:
                loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=self.pos_class_weight)
            else:
                loss = None

            return {
                "loss": loss,
                "logits": logits,
                "probs": probs,
                "demo_contrib": demo_contrib,
                "obs_contrib": obs_contrib,      # (B, N)
                "att_weights": att,              # (B, N)
                "ts_contrib": ts_contrib         # (B,)
            }