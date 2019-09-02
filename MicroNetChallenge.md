# MicroNet Challenge

by Mengze Zeng(曾梦泽), Jie Hu(胡杰),  Ziheng Wu(吴梓恒) from [Momenta](https://www.momenta.ai/)

[TOC]



## 1 Methodology

### 1.1 BaseModel

### 1.2 Sparsity

Non-structured pruning is an efficient way to reduce the computing and storage requirements of convolutional neural network in the future。

#### 1.2.1 Han's Strategy(DSD)

Acorrding to Han's proposal, dense-sparse-dense training flow is a  novel training strategy which will keep maintain or even improve the neural network classification accuracy after iteratively sparse pruning and retraining process.(as figure.1)

![figure.1 dsd](figures/dsd.png)

#### 1.2.2 Implementation

Implementation of DSD(dense-sparse-dense) strategy with caffe is used to pruning our proposal network.  DSD pipeline includes:

- Sensitivity Analysis:  [commit](https://github.com/Zeitzmz/micronet-caffe/commit/d42e729e2232777b28eeb6446d7f7d2e58ea1d79)
- GenerateMask :        [commit](https://github.com/Zeitzmz/micronet-caffe/commit/f8303ac4b56bc74a832d35d8f69742345eb54c9e)
- Pruning & Retrain:    [commit](https://github.com/Zeitzmz/micronet-caffe/commit/f8303ac4b56bc74a832d35d8f69742345eb54c9e)

##### Sensitivity Analysis 

commit info: [commit](https://github.com/Zeitzmz/micronet-caffe/commit/d42e729e2232777b28eeb6446d7f7d2e58ea1d79)

According to Han's proposal, pruning procedure is to replace the weights which has smaller absolute value than others in the same layer with zero. This proposal is kind of heuristic but effective.   Sensitivity Analysis is a method to decide which layers needed to be pruned and the sparsity of these layers after pruning.  Our implemented tool allows us to:

- Assign the layer to be pruned and percentage of sparsity for a trained model, and test the accuray of the pruned model.

With traversing the accuracy of different sparsity and layer combination with in our target layer with python or shell scripts, we will get a sensitivity map as bellow:

![dsd_result](figures/dsd_result.png)

With this map, we will find which layers are important and can't be  over pruned。Heuristiclly, we choose some 'unimportant' layers with their proposal prune ratio  and generate a config file.

##### GenerateMask

commit info:  [commit](https://github.com/Zeitzmz/micronet-caffe/commit/f8303ac4b56bc74a832d35d8f69742345eb54c9e)

Sensitivity analysis generates a config file which records the layers name and sparsity. This tool is to provide the zero mask which will be needed during caffe infer (after final prune) and DSD retrain.

- Assign the model file and config file , generate the zero-one maskfile.

##### Pruning & Retrain

Commit info:  [commit](https://github.com/Zeitzmz/micronet-caffe/commit/f8303ac4b56bc74a832d35d8f69742345eb54c9e)

Our modified caffe support to finetune the sparse layer which will keep the sparsity during finetune.

- Forward: mask the sparse part with input mask, and forward.
- Backword:  backword normally, and mask the diff , then update weights. zero part will keep zero during the finetune. 

During  our experiments we noticed that iteratively DSD and Re-Dense Part is hardly to improve the sparsity of our final model with target accuracy limitation, so we just stop at the retrain part with mask which not more future re-densing and pruning.

**With this incomplete dsd approach(sensitivity_analysis - prune - retain_with_mask), we're able to prune our trained model at 30~50% sparsity with slightly loss in accuracy.** 

### 1.3 Quantization

Quantization is a more sophisticated technology then sparsification in deep learning. With Google famous white paper \<Quantizing deep convolutional networks for efficient inference: A whitepaper\> [arxiv](https://arxiv.org/pdf/1806.08342.pdf) and Nvidia official TensorRT int8 support [INT8 Inference](http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf) , quantization has become an essential part of a deep learning model.

We have to mention that our quantization is **fake quantization** according to the [micrionet-challenge-scoring](https://micronet-challenge.github.io/scoring_and_submission.html) with FP16 accumulation.

#### 1.3.1 Quantization method

Our proposal model utilized three different quantization for different parts:

- Symmetric KL-divergence based quantization
- Asymmetric KL-divergence based quantization
- Symmetric *MaxValue* quantization

Quantization for FP32 vectors ***X***  is to find "best" FP32 step *s*, and do the transform below :
$$
x_{int} = clamp((round(\frac{x}{s}), min_{int}, max_{int}))
$$

##### 1.3.1.1 Symmetric KL-divergence based quantization

 Symmetric KL-divergence base quantization is used for featuremap which has negative response, such as Inputdata, Convolution Output without ReLU.

**Hyperparameter**

- precision : define the bits of int format, such as int8 means precision=8.
  $$
  \begin{aligned}
  INT_{max} &= 2^{precision-1}-1\\
  INT_{min} &= -2^{precision-1}
  \end{aligned}
  $$
  
- tolerance:  scale factor threshold for min kl-divergence.  Usually, the min kl-distance's distribution(step) doesn't generate the best quantization model, we introduce this hypermeter to relax the limitaion. Instead, all distribution(step) which has kl-divergence  less then *tolerance* * min(kl) are considered, we always find the distribution with max step.

**Pseudo-code of getting s**

limitation: precision<=10

```c++
GenerateHist(fabs(Input))           # output as 2048 bins 
Input:  FP32 histogram H with 2048 bins: bin[ 0 ], …, bin[ 2047 ]
For i in range( (max_{int} + 1) , 2048 ):
    reference_distribution_P = [ bin[ 0 ] , ..., bin[ i-1 ] ] // take first ‘ i ‘ bins from H
    outliers_count = sum( bin[ i ] , bin[ i+1 ] , … , bin[ 2047 ] )
    reference_distribution_P[ i-1 ] += outliers_count
    P /= sum(P) // normalize distribution P
    candidate_distribution_Q = quantize [ bin[ 0 ], …, bin[ i-1 ] ] into 128 levels explained later
    expand candidate_distribution_Q to ‘ i ’ bins // explained later
    Q /= sum(Q) // normalize distribution Q
    divergence[ i ] = KL_divergence( reference_distribution_P, candidate_distribution_Q)
End For

Find the minimal KL_divergence called minKL
Find the max index 'h' for which divergence[ k ] <= tolerance * minKL 
s = ( h + 0.5 ) * ( width of a bin ) / (max_{int} + 1)
return s
```
*Here is the diagram for precision=8*
![symmetric_kl](/Users/wuziheng/Desktop/september/nips-micronet/figures/symmetric_kl.png)

##### 1.3.1.2 Asymmetric KL-divergence based quantization

This is almost the same as Symmetric KL-divergence based quantization, except  for layers which only generate positive response, such as convolution-bn-relu combination. As the figure below, we ignore the negative aixs representation(that we don't need) which economize one bit compared to Symmetric KL-divergence based quantization.
$$
\begin{aligned}
INT_{max} &= 2^{precision}-1\\
INT_{min} &= 0
\end{aligned}
$$
*Here is the diagram for precision=8*

![asymmetric_kl](/Users/wuziheng/Desktop/september/nips-micronet/figures/asymmetric_kl.png)



##### 1.3.1.3 Symmetric *MaxValue* quantization

Empirically, we use this Symmetric *MaxValue* quantization for convolution kernels, and channel-wise quantization is also adopted for further improving accuracy.
$$
\begin{aligned}
INT_{max} &= 2^{precision-1}-1\\
INT_{min} &= -2^{precision-1}\\
\end{aligned}
$$

$$
s = \frac{max(fabs(X))}{2^{precision-1}-1}
$$

*Here is the diagram for precision=8*
![maxvalue](/Users/wuziheng/Desktop/september/nips-micronet/figures/maxvalue.png)

#### 1.3.2 Accumulation with FP16

For common implementation of quantized matrix dot product, the accumulation part is still FP32 or INT32 to keep the performance.   Here is the **cudnn** implementation which use INT32 accumulation.

![cudnn_conv](/Users/wuziheng/Desktop/september/nips-micronet/figures/cudnn_conv.png)

In order to further accelerate accumulation process, we find that FP16 is accurate enough in modern neural network with normalization technology (e.g. BatchNorm, GroupNorm, ...). In this repo, we use cublas engine with FP16 dataType and computeType to conduct GEMM in Convolution operations.



## Training & Testing



## Scoring

