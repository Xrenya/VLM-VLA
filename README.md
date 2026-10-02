# VLM/VLA Tutorial
0. 
1. [Introduction VLM notebook](https://github.com/Xrenya/VLM-VLA/VLM_1.ipynb): short introduction into how VLM training is done with the simple examples:
   -  VLM training stage for different models (about attention and MMDiT arch see [repo](https://github.com/Xrenya/stable_diffusion)
   -  Sample Packing (Multipack)
   -  4D mask for attention and RoPE embeddings correct calculation
   -  Inference of the finetuned model (mentioned importance of usage KV-cache, can be found in this [tutorial](https://github.com/Xrenya/pytorch-seq2seq/blob/main/7_KVCaching.ipynb))

The model was trained on 491M tokens, but overall make decent predictions.  
**Example**:  
![uniqlo](assets/uniqlo.jpeg)

**Generate output**:
> [!NOTE]
> The image displays an indoor setting with a large, open-air display featuring a variety of clothing items.
> The display is positioned in front of a large, wooden sign with the text "UNIDO" in white letters,
> which is surrounded by a green background. The sign is positioned on a wooden floor,
> and the sign is adorned with a green and white banner with the text "UNIDO" in white letters.

Model weights: 🤗[Xrenya/smolVLM](https://huggingface.co/Xrenya/smolVLM)

2. Introduction VLA with Linear projection heads notebook
3. Introduction VLA with diffusion head notebook
