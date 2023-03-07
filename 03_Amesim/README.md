# fmudiff

## Description

`fmudiff` extends [`torch.autograd`](
https://pytorch.org/docs/stable/notes/extending.html) (PyTorch's AD framework) so that
FMU's can be integrated in PyTorch gradient descent optimization problems, allowing to
define hybrid models that combine physics and neural networks.  
This folder contains an example with an Amesim-generated FMU, representing a Van der Pol
oscillator.

## Acknowledgement

The approach is inspired by [FMIFlux.jl](https://github.com/ThummeTo/FMIFlux.jl); also
refer to:  
  
Tobias Thummerer, Lars Mikelsons and Josef Kircher. 2021. **NeuralFMU: towards
structural integration of FMUs into neural networks.** Martin Sjölund, Lena Buffoni,
Adrian Pop and Lennart Ochel (Ed.). Proceedings of 14th Modelica Conference 2021,
Linköping, Sweden, September 20-24, 2021. Linköping University Electronic Press,
Linköping (Linköping Electronic Conference Proceedings ; 181), 297-306. [DOI:
10.3384/ecp21181297](https://doi.org/10.3384/ecp21181297)
