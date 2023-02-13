# DRVN (deep random vortex network)

This repository contains the code for the paper
- [DRVN (deep random vortex network): A new physics-informed machine learning method for simulating and inferring incompressible fluid flows](https://aip.scitation.org/doi/abs/10.1063/5.0110342)

We present the deep random vortex network (DRVN), a novel physics-informed framework for simulating and inferring the fluid dynamics governed by the incompressible Navier–Stokes equations. Unlike the existing physics-informed neural network (PINN), which embeds physical and geometry information through the residual of equations and boundary data, DRVN automatically embeds this information into neural networks through neural random vortex dynamics equivalent to the Navier–Stokes equation. Specifically, the neural random vortex dynamics motivates a Monte Carlo-based loss function for training neural networks, which avoids the calculation of derivatives through auto-differentiation. Therefore, DRVN can efficiently solve Navier–Stokes equations with non-differentiable initial conditions and fractional operators. Furthermore, DRVN naturally embeds the boundary conditions into the kernel function of the neural random vortex dynamics and, thus, does not need additional data to obtain boundary information. We conduct experiments on forward and inverse problems with incompressible Navier–Stokes equations. The proposed method achieves accurate results when simulating and when inferring Navier–Stokes equations. For situations that include singular initial conditions and agnostic boundary data, DRVN significantly outperforms the existing PINN method. Furthermore, compared with the conventional adjoint method when solving inverse problems, DRVN achieves a 2 orders of magnitude improvement for the training time with significantly precise estimates.

## Quick Start

To train DRVN on Lamb-Oseen vortex, use
```bash
python main.py
```

## Citation

If you find our work useful in your research, please consider citing:

```
@article{zhang2022drvn,
  title={DRVN (deep random vortex network): A new physics-informed machine learning method for simulating and inferring incompressible fluid flows},
  author={Zhang, Rui and Hu, Peiyan and Meng, Qi and Wang, Yue and Zhu, Rongchan and Chen, Bingguang and Ma, Zhi-Ming and Liu, Tie-Yan},
  journal={Physics of Fluids},
  volume={34},
  number={10},
  pages={107112},
  year={2022},
  publisher={AIP Publishing LLC}
}
```
