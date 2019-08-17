* Paper  
  * Title: Self-Attention Generative Adversarial Networks 
  
  * Authors: Han Zhang, Ian Goodfellow, Dimitris Metaxas, Augustus Odena  
  
  * Link: https://arxiv.org/pdf/1805.08318.pdf 
  
  * Year: 2018  
  
  * Code:  https://github.com/brain-research/self-attention-gan, https://github.com/heykeetae/Self-Attention-GAN
  
* Summary  
  * What:  
    * In this paper authors introduce a novel type of attention mechanism for GANs. They develop self-attention layer that is aimed to solve certain problems that GANs run into, particularly the GANs' inability to generate high quality images that contain structural patterns (like dog's feet). The proposed network achieves better success in modeling long-range multi-level dependencies in images. Authors  also illustrate how the network utilizes attention mechanism to model patterns that occur in images.  
  * How:  
    * Self-attention layer is implemented by transforming image features from previous layer into two feature spaces `f` and `g`. Then attention map is calculated by applying softmax on each row of the matrix `A = f^T * g`. Attention map is multiplied by another transformation of the input `h`. Result is in turn transformed with multiplication by learnable weight matrix producing self-attention feature maps. Weight matrices for this layer are implemented as 1x1 convolutions. Final output of the layer is in form `gamma * o + x`, where `o` is the current output, `x` is the original input and `gamma` is learnable scalar initialized to zero. This is made to allow network to rely on local neighbourhood firstly and gradually increase weight of non-local features. Self-attention modules were embedded into generator and discriminator in various positions and compared with embedding residual blocks. Training is performed by minimizing the hinge version of adversarial loss:
  ![alt text](https://github.com/pgmvp/ComputerVision/blob/master/Homework5/images/loss.png "Loss function")  
     * How does the self-attention layer look like:  
  ![alt text](https://github.com/pgmvp/ComputerVision/blob/master/Homework5/images/attn.png "Self-attention layer")  
     * Authors use two recent techniques to stabilize training of GAN. First is spectral normalization. Normalization is attained by constraining the spectral norm of the layers' weight matrices to be 1. This normalization is applied  both to generator and discriminator (in earlier works it was applied only to discriminator).
     * Second technique is TTUR (two-timescale update rule). This means using different learning rates for discriminator and generator. A recent paper has shown that networks trained in this fashion converge to local Nash equilibrium. This technique also reduces the number of updates needed for discriminator per generator update.
  
  * Results:  
    * Authors evaluate their architerture on ImageNet dataset. Evaluation metrics are Inception score and Frechet Inception distance. They report an increase in best published Inception score from 36.8 to 52.52 and reducing the best known Frechet Inception distance from 27.62 to 18.65.  
    * They further evaluate stabilizing techniques for the model by comparing it with a baseline (state-of-the-art image generation model). SAGAN with spectral normalization and with spectral normalization + TTUR is considered and shown that the last option outperforms baseline and first option:
![alt text](https://github.com/pgmvp/ComputerVision/blob/master/Homework5/images/results.png "Evaluation")  
    * Authors also evaluate the self-attention mechanism by plugging the proposed layer in different places of the network and comparing its results with the network where residual blocks were embedded (`feat_k` means adding self-attention to the `k x k` feature maps):  
![alt text](https://github.com/pgmvp/ComputerVision/blob/master/Homework5/images/table.png "Table")  
    * It is shown that self-attention outperforms both baseline and embedding of the residual blocks.
