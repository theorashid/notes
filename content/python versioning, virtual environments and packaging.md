---
tags:
  - install
  - python
  - swe
folder: learning
title: python versioning, virtual environments and packaging
date created: Monday, January 29th 2024, 7:01:22 pm
date modified: Sunday, February 25th 2024, 12:19:12 pm
share: "true"
---

The main resource is Anna-Lena Popkes' post [_An unbiased evaluation of environment management and packaging tools_](https://alpopkes.com/posts/python/packaging_tools/).

## python virtual environments

`pyenv` is the main python version management tool, which makes it easy to switch between multiple versions of Python. It works alongside a virtual environment manager `venv` (or `virtualenv`) and a package installer `pip`.

`conda` is its own distribution of python (and R), and contains the python version and packages all in one, as well as dealing with package management. It is commonplace in scientific computing.

## pyenv + venv + pip

```sh
pyenv local 3.11 # creates local file .python-version
python3 -m venv .venv # creates local folder venv/ with all packages
source .venv/bin/activate
pip install -e ."[dev]"
```

## mamba (conda, but quick)

```sh
mamba create -n new-env python=3.11
mamba activate new-env
pip install -e ."[dev]"
```

This worked for me for a long time. I tried to move to micromamba, which does not have a base environment, but [VSCode would not let me select micromamba environments](https://github.com/microsoft/vscode-python/issues/20919) as the python interpreter.

I am excited for the [`pixi`](https://github.com/prefix-dev/pixi) project, which is being built by the mamba devs. It aims to overcome many of the issues whilst also being a multi-language package manager. You will be able to build `pip` and `conda` packages using `pixi`.

## python package management

[Everything should get a python package](https://ericmjl.github.io/blog/2022/3/31/everything-gets-a-package-yes-everything-gets-a-package/). We should constantly be refactoring model development notebooks into `.py` files.

However, python packaging is a [mess](https://chriswarrick.com/blog/2023/01/15/how-to-improve-python-packaging/).

I am periodically trying different tools (listed on [PyPA](https://packaging.python.org/en/latest/key_projects/)) using the latest practices. Use a `pyproject.toml`, not `setup.py`.

Below are some working notes:

| package | comments | example repo |
| ---- | ---- | ---- |
| setuptools | Default |  |
| hatch | Not bad, although need to enter dependencies manually. | [tinygp](https://github.com/dfm/tinygp) |
| flit | Why aren't more people using this? Hardly any projects are not pure python. This simplifies things massively. | [bayeux](https://github.com/jax-ml/bayeux) |
| poetry | I've used it a lot. It's easy but it's kind of slow... to resolve dependencies. | [GPJax](https://github.com/JaxGaussianProcesses/GPJax) |
| pdm | Not yet used |  |
| uv | drop in replacement for pip (and soon everything else). rapid | [uv](https://github.com/astral-sh/uv) |
