---
tags:
  - install
  - python
  - swe
folder: learning
title: python versioning, virtual environments and packaging
date created: Monday, January 29th 2024, 7:01:22 pm
date modified: Monday, September 9th 2024, 11:54:24 pm
share: true
---

The main resource is Anna-Lena Popkes' post [_An unbiased evaluation of environment management and packaging tools_](https://alpopkes.com/posts/python/packaging_tools/). See also [_Modern good practices for python development_.](https://www.stuartellis.name/articles/python-modern-practices/)

## python virtual environments

`pyenv` is a python version management tool, which makes it easy to switch between multiple versions of Python. It works alongside a virtual environment manager `venv` (or `virtualenv`) and a package installer `pip`. The [`uv` project](https://docs.astral.sh/uv/) aims to replace all of these.

`conda` is its own distribution of python (and R), and contains the python version and packages all in one, as well as dealing with package management. It is commonplace in scientific computing.

### uv

Replacing `pyenv + venv + pip`

```sh
uv python install 3.11 3.12 # install multiple versions of python, correct version specified in pyproject.toml
```

```sh
uv venv # creates .venv/ virtual environment with all the packages
source .venv/bin/activate
uv pip install -e ."[dev]"
```

### mamba (conda, but quick)

```sh
mamba create -n new-env python=3.11
mamba activate new-env
pip install -e ."[dev]"
```

This worked for me for a long time. I tried to move to micromamba, which does not have a base environment, but [VSCode would not let me select micromamba environments](https://github.com/microsoft/vscode-python/issues/20919) as the python interpreter.

Time to [try out](https://ericmjl.github.io/blog/2024/8/16/its-time-to-try-out-pixi/) [`pixi`](https://github.com/prefix-dev/pixi) project, which is built by the mamba devs. It aims to overcome many of the issues whilst also being a multi-language package manager. You will be able to build `pip` and `conda` packages using `pixi`. It has VSCode support.

## python package management

[Everything should get a python package](https://ericmjl.github.io/blog/2022/3/31/everything-gets-a-package-yes-everything-gets-a-package/). We should constantly be refactoring model development notebooks into `.py` files.

However, python packaging ~~is~~ was a [mess](https://chriswarrick.com/blog/2023/01/15/how-to-improve-python-packaging/). Now `uv` will [sort everything out](https://docs.astral.sh/uv/guides/projects/).

```sh
mkdir <project name>
cd <project name>
uv init --lib # creates pyproject.toml and uses hatch for packaging
uv add <package> # creates uv.lock
uv add <dev package> --optional <group> # adds to dev-dependecies in pyproject
# OR
uv pip install # pip interface
uv add git+https://github.com/encode/httpx # git dependencies
# use uv sync or uv venv to make venv
uv run <script name>.py
```

I am periodically trying different tools (listed on [PyPA](https://packaging.python.org/en/latest/key_projects/)) using the latest practices. Use a `pyproject.toml`, not `setup.py`.

Below are some working notes:

| package    | comments                                                                                                            | example repo                                           |
| ---------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| setuptools | Default                                                                                                             |                                                        |
| hatch      | Not bad, although need to enter dependencies manually (unless using `uv`).                                          | [tinygp](https://github.com/dfm/tinygp)                |
| flit       | Why aren't more people using this? Hardly any projects are not pure python. This simplifies things massively.       | [bayeux](https://github.com/jax-ml/bayeux)             |
| poetry     | I've used it a lot. It's easy but it's kind of slow... to resolve dependencies.                                     | [GPJax](https://github.com/JaxGaussianProcesses/GPJax) |
| pdm        | Not yet used                                                                                                        |                                                        |
| uv         | [drop in replacement for pip](https://astral.sh/blog/uv-unified-python-packaging) (and soon everything else). rapid | [uv](https://github.com/astral-sh/uv)                  |

## standalone scripts

We can [declare the dependencies](https://www.youtube.com/watch?v=jXWIxk2brfk) for a script in the script itself [via `uv`](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies).
