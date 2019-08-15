Paper  
  Title: AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks  
  
  Authors: Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, Xiaodong He  
  
  Link: http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf  
  
  Tags:  
  
  Year: 2018  
  
  Code:  
  
Summary  
What:  
  In this paper authors introduce a novel type of attention mechanism for GANs. They develop self-attention layer that is aimed to solve certain problems that GANs run into, particularly the GANs' inability to generate high quality images that contain structural patterns (like dog's feet). The proposed network achieves better success in modeling long-range multi-level dependencies in images. Authors  also illustrate how the network utilizes attention mechanism to model patterns that occur in images.  
  How:  
  Self-attention module is embedded into generator and discriminator. It calculates response at a position as a weighted sum of the features at all positions. Weight matrices for this layer are implemented as 1x1 convolutions.  
![alt text](https://github.com/pgmvp/ComputerVision/blob/master/Homework5/images/attn.png "Self-attention layer")
  
  Results:  



orange - batch norm
white - relu
light yellow - attention
