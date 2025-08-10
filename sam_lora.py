# Sheng Wang at Apr 6 2023
# What a time to be alive (first half of 2023)

from SAMraw import build_sam, SamPredictor
from SAMraw import sam_model_registry
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from SAMraw.modeling import Sam
from safetensors import safe_open
from safetensors.torch import save_file

import pytorch_lightning as pl
from typing import Any, Dict, List, Tuple
import numpy as np

from utils import eval_seg
from losses import SamDiceLoss


class _LoRA_qkv(pl.LightningModule):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_k: nn.Module,
        linear_b_k: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        q=True,
        k=True,
        v=True,
    ):
        super().__init__()
        self.qkv = qkv
        self.q = q
        self.k = k
        self.v = v
        if self.q:
            self.linear_a_q = linear_a_q
            self.linear_b_q = linear_b_q
        if self.k:
            self.linear_a_k = linear_a_k
            self.linear_b_k = linear_b_k
        if self.v:
            self.linear_a_v = linear_a_v
            self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.in_features = qkv.in_features
        self.out_features = qkv.out_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        if self.q:
            new_q = self.linear_b_q(self.linear_a_q(x))
            qkv[:, :, :, : self.dim] += new_q
        if self.k:
            new_k = self.linear_b_k(self.linear_a_k(x))
            qkv[:, :, :, self.dim:-self.dim] += new_k
        if self.v:
            new_v = self.linear_b_v(self.linear_a_v(x))
            qkv[:, :, :, -self.dim :] += new_v
        return qkv

class _LoRA_output(pl.LightningModule):

    def __init__(
        self,
        proj: nn.Module,
        linear_a_proj: nn.Module,
        linear_b_proj: nn.Module,
    ):
        super().__init__()
        self.proj = proj
        self.linear_a_proj = linear_a_proj
        self.linear_b_proj = linear_b_proj
        self.dim = proj.in_features
        self.in_features = proj.in_features
        self.out_features = proj.out_features
        self.w_identity = torch.eye(proj.in_features)

    def forward(self, x):
        proj = self.proj(x)  
        new_proj = self.linear_b_proj(self.linear_a_proj(x))
        proj += new_proj
        return proj

class LoRA_Sam(pl.LightningModule):
    """Applies low-rank adaptation to a Sam model's image encoder.

    Args:
        sam_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, sam_model: Sam, r: int, args, q=True,k=True,v=True,out=True,lora_layer=None):
        super(LoRA_Sam, self).__init__()

        assert r > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.blocks)))
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            w_proj_linear = blk.attn.proj
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_k = nn.Linear(self.dim, r, bias=False)
            w_b_linear_k = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            w_a_linear_proj = nn.Linear(self.dim, r, bias=False)
            w_b_linear_proj = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_k)
            self.w_Bs.append(w_b_linear_k)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            self.w_As.append(w_a_linear_proj)
            self.w_Bs.append(w_b_linear_proj)
            blk.attn.qkv = _LoRA_qkv(
                qkv=w_qkv_linear,
                linear_a_q=w_a_linear_q,
                linear_b_q=w_b_linear_q,
                linear_a_k=w_a_linear_k,
                linear_b_k=w_b_linear_k,
                linear_a_v=w_a_linear_v,
                linear_b_v=w_b_linear_v,
                q=q,
                k=k,
                v=v,
            )
            if out:
                blk.attn.proj = _LoRA_output(
                    proj=w_proj_linear,
                    linear_a_proj=w_a_linear_proj,
                    linear_b_proj=w_b_linear_proj,
                )
        self.reset_parameters()
        self.sam = sam_model
        self.loss = SamDiceLoss()
        self.args = args
        self.threshold = (0.1, 0.3, 0.5, 0.7, 0.9)

    def load_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")
        _in = self._LoRA_qkv.head.in_features
        _out = self._LoRA_qkv.head.out_features
        with safe_open(filename, framework="pt") as f:
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self._LoRA_qkv.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        
        save both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        
        _in = self.sam.image_encoder.blocks[0].attn.qkv.in_features
        _out = self.sam.image_encoder.blocks[0].attn.qkv.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.sam.image_encoder.blocks[0].attn.qkv.identity.weight}
        
        merged_dict = {**a_tensors, **b_tensors, **fc_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\
            
        load both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)
                
            _in = self.lora_vit.head.in_features
            _out = self.lora_vit.head.out_features
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
            #nn.init.normal_(w_A.weight)
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward_batch(
        self,
        batched_input: List[Dict[str, Any]],
    ) -> torch.Tensor:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        imgs = batched_input['image'].to(dtype = torch.float32, device = self.device)
        pt = batched_input['pt']
        point_labels = batched_input['p_label']
        point_coords = pt
        coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
        labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
        bbox_torch = torch.as_tensor(batched_input['bbox'], dtype=torch.float, device=self.device)
        lowmask_torch = torch.as_tensor(batched_input['low_mask'], dtype=torch.float, device=self.device)


        '''
        Reduce Control Pts with Epoch
        '''

        if self.current_epoch > self.args.epoch//4:
          epochRatio = (self.current_epoch+1)/(self.args.epoch)
          numSelectPts = int(np.round(20*(1-epochRatio)))
          if numSelectPts<2:
              numSelectPts = 2
          _,mPts,_ = coords_torch.shape
          rPts = np.random.choice(mPts,numSelectPts,replace=False)
          coords_torch = coords_torch[:,rPts,:]
          labels_torch = labels_torch[:,rPts]

        pt = (coords_torch, labels_torch)


        imge = self.sam.image_encoder(imgs)

        if np.random.rand()<0.3:
            pt = None
        if np.random.rand()<0.5:
            bbox_torch = None
        if np.random.rand()<0.4:
            lowmask_torch = None

        se, de = self.sam.prompt_encoder(
            points=pt,
            boxes=bbox_torch,
            masks=lowmask_torch,
        )
        
        pred, _ = self.sam.mask_decoder(
          image_embeddings=imge,
          image_pe=self.sam.prompt_encoder.get_dense_pe(), 
          sparse_prompt_embeddings=se,
          dense_prompt_embeddings=de, 
          multimask_output=False,
        )
        
        pred = F.interpolate(
          pred,
          size=(self.args.image_size, self.args.image_size),
          mode="bilinear",
          align_corners=False,
        )
        return pred

    def forward_predict(
        self,
        batched_input: List[Dict[str, Any]],
    ) -> torch.Tensor:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        imgs = batched_input['image'].to(dtype = torch.float32, device = self.device)
        pt = batched_input['pt']
        point_labels = batched_input['p_label']
        point_coords = pt
        coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
        labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
        bbox_torch = torch.as_tensor(batched_input['bbox'], dtype=torch.float, device=self.device)
        lowmask_torch = torch.as_tensor(batched_input['low_mask'], dtype=torch.float, device=self.device)




        rPts = 10
        coords_torch = coords_torch[:,:rPts,:]
        labels_torch = labels_torch[:,:rPts]

        pt = (coords_torch, labels_torch)

        # pt = None
        bbox_torch = None

        lowmask_torch = None


        imge = self.sam.image_encoder(imgs)

        se, de = self.sam.prompt_encoder(
            points=pt,
            boxes=bbox_torch,
            masks=lowmask_torch,
        )
        
        pred, _ = self.sam.mask_decoder(
          image_embeddings=imge,
          image_pe=self.sam.prompt_encoder.get_dense_pe(), 
          sparse_prompt_embeddings=se,
          dense_prompt_embeddings=de, 
          multimask_output=False,
        )
        
        pred = F.interpolate(
          pred,
          size=(self.args.image_size, self.args.image_size),
          mode="bilinear",
          align_corners=False,
        )
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p : p.requires_grad, self.parameters()), lr=self.args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        self.lr_schedulers = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epoch)
        return optimizer
    
    def training_step(self,train_batch, batch_idx):
        pred = self.forward_batch(train_batch)
        masks = train_batch['label'].to(dtype = torch.float32, device = pred.device)

        loss    = self.loss(pred, masks)
        # print(loss)
        eiou, edice = eval_seg(pred,masks,self.threshold)

        self.log('train_loss',loss)
        self.log('train_eiou',eiou)
        self.log('train_edice',edice)
        return loss

    def validation_step(self,valid_batch,batch_idx):
        pred = self.forward_batch(valid_batch)
        masks = valid_batch['label'].to(dtype = torch.float32, device = pred.device)

        loss    = self.loss(pred, masks)
        eiou, edice = eval_seg(pred,masks,self.threshold)

        self.log('valid_loss',loss)
        self.log('valid_eiou',eiou)
        self.log('valid_edice',edice)
        return loss

    def on_train_epoch_end(self):
        sch = self.lr_schedulers
        sch.step()


if __name__ == "__main__":
    args = argparse.Namespace()
    args.image_size = 256
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    args.encoder_adapter = True
    # args.sam_checkpoint = "pretrain_model/sam_vit_b.pth"
    args.sam_checkpoint = "workdir/models/sam-med2d/epoch30_sam.pth"
    model = sam_model_registry["vit_b"](args).to(device)
    lora_sam = LoRA_Sam(model,16).to(device)
    with open(args.sam_checkpoint, "rb") as f:
        state_dict = torch.load(f)
        lora_sam.sam.load_state_dict(state_dict['model'])
    print(lora_sam.sam)
    for n, value in lora_sam.sam.named_parameters():
        print(n,value.requires_grad)
    
    # lora_sam.sam.image_encoder(torch.rand(size=(1,3,256,256)).to(device))