# Conda env setup

This should cover the basic dependencies.

```conda create --name blackjax "python<=3.10" blackjax pymc jax jaxlib ipython jupyter notebook```

If you encounter an issue with missing `crypt.h` install `libxcrypt` via conda and set a symlink in


```ln -s <path>/<to>/miniforge3/envs/blackjax/include/crypt.h <path>/<to>/miniforge3/envs/blackjax/include/python3.10/crypt.h```

