# AutoencoderPyTorch
A comprehensive tutorial on how to implement various autoencoder models using PyTorch

## Demo notebooks
- [TrainSimpleFCAutoencoder](./TrainSimpleFCAutoencoder.ipynb) notebook demonstrates how to implement and train very simple a fully-connected autoencoder with a single-layer encoder and a single-layer decoder.
- [TrainDeepSimpleFCAutoencoder](./TrainDeepSimpleFCAutoencoder.ipynb) and [TrainDeeperSimpleFCAutoencoder](./TrainDeeperSimpleFCAutoencoder.ipynb) notebooks demonstrate how to implement and train a fully-connected autoencoder with a multi-layer encoder and a multi-layer decoder. Please first train single-layer autoencoder using the [TrainSimpleFCAutoencoder](./TrainSimpleFCAutoencoder.ipynb) notebook as the very initial pretrain model for the deeper autoencoder training notebooks. Then, gradually increase depth of the autoencoder and use previously trained (shallower) autoencoder as the pretrained model.
- [TrainSimpleConvAutoencoder](./TrainSimpleConvAutoencoder.ipynb) notebook demonstrates how to implement and train an autoencoder with a convolutional encoder and a fully-connected decoder.
- [TrainSimpleSparseFCAutoencoder](./TrainSimpleSparseFCAutoencoder.ipynb) notebook demonstrates how to implement and train an autoencoder with hard (feature) sparsity and lifetime (winner takes all) sparsity.
- [TrainSimpleFCAutoencoderWithSparseLoss](./TrainSimpleFCAutoencoderWithSparseLoss.ipynb) notebook demonstrates how to implement and train autoencoder with soft sparsity loss.
- [TrainSimpleDenoiseFCAutoencoder](./TrainSimpleDenoiseFCAutoencoder.ipynb) notebook demonstrates how to implement and train an autoencoder for denoise applications.
- Notebooks with name starting with "Plot*" are used to plot the training results.

## Tutorial
A step by step tutorial on how to build and train VGG using PyTorch can be found in my [blog post](https://jianzhongdev.github.io/VisionTechInsights/posts/autoencoders_with_pytorch_full_code_guide/) (URL: https://jianzhongdev.github.io/VisionTechInsights/posts/autoencoders_with_pytorch_full_code_guide/ ) 

## Dependency
This repo has been implemented and tested on the following dependencies:
- Python 3.10.13
- matplotlib 3.8.2
- numpy 1.26.2
- torch 2.1.1+cu118
- torchvision 0.16.1+cu118
- notebook 7.0.6

## Computer requirement
This repo has been tested on a laptop computer with the following specs:
- CPU: Intel(R) Core(TM) i7-9750H CPU
- Memory: 32GB 
- GPU: NVIDIA GeForce RTX 2060

## License

[GPL-3.0 license](./LICENSE)

## Reference

[1] Hinton, G. E. & Salakhutdinov, R. R. Reducing the Dimensionality of Data with Neural Networks. Science 313, 504–507 (2006).

[2] Kramer, M. A. Nonlinear principal component analysis using autoassociative neural networks. AIChE Journal 37, 233–243 (1991).

[3] Masci, J., Meier, U., Cireşan, D. & Schmidhuber, J. Stacked Convolutional Auto-Encoders for hierarchical feature extraction. in Lecture notes in computer science 52–59 (2011). doi:10.1007/978-3-642-21735-7_7.

[4] Makhzani, A. & Frey, B. J. A Winner-Take-All method for training sparse convolutional autoencoders. arXiv (Cornell University) (2014).

[5] A. Ng, “Sparse autoencoder,” CS294A Lecture notes, vol. 72, 2011.

## Citation

If you found this article helpful, please cite it as:
> Zhong, Jian (June 2024). Autoencoders with PyTorch: Full Code Guide. Vision Tech Insights. https://jianzhongdev.github.io/VisionTechInsights/posts/autoencoders_with_pytorch_full_code_guide/.

Or

```html
@article{zhong2024buildtrainAutoencoderPyTorch,
  title   = "Autoencoders with PyTorch: Full Code Guide",
  author  = "Zhong, Jian",
  journal = "jianzhongdev.github.io",
  year    = "2024",
  month   = "June",
  url     = "https://jianzhongdev.github.io/VisionTechInsights/posts/autoencoders_with_pytorch_full_code_guide/"
}
```