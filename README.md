# Extending the Lifetime of Broken Mobile Phone Lense through Deconvolution

This repository includes three different non-blind deconvolution approaches: Wiener Deconvolution (due to CITE), Total Variation Deconvolution for a spatially-invariant PSF through ADMM (due to CITE), and Total Variation Deconvolution for a spatially-variant PSF throguh the Douglas-Rachford algorithm (due to CITE).

These can be found in the `admm.py` and `douglas_rachford.py` files, with the Wiener deconvolution results being computed directly in `wiener.ipynb` notebook using skimage's built in method.

My Phone's PSF calibration can be found in the `psf_calib` directory alongside a jupyter notebook used to process them. The results from using them with deconvolution can be found in the `phone_experiments.ipynb` notebook.

Presentation: https://drive.google.com/file/d/1G-NlzhEQiuJ2izplj2pNZrekV6bw8Kw9/view?usp=sharing
