---
tags:
  - install
  - mac
  - r
  - python
  - quarto
  - swe
folder: learning
title: mac setup
date created: Monday, January 29th 2024, 7:01:22 pm
date modified: Friday, December 26th 2025, 10:54:11 am
share: true
---

## Basics

1. Install [homebrew](https://brew.sh) for package management.
2. Replace terminal with [iTerm2](https://formulae.brew.sh/cask/iterm2).
3. [oh-my-zsh](https://ohmyz.sh/) to configure zsh.
4. `defaults write com.apple.finder AppleShowAllFiles YES`
5. Connect [GitHub account with ssh](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)
`brew install`:
- `git` 
- [`eza`](https://github.com/eza-community/eza/blob/main/INSTALL.md) to make `ls` and `tree` prettier. [Eric Ma's essay](https://ericmjl.github.io/essays-on-data-science/terminal/cli-tools/) has a list of other useful terminal hacks.
	- Add `alias ls='eza --long --git --header --group'` and `alias tree='eza --tree --level=2 --long --header --git'` to `.zshrc`
- [`imagemagick`](https://formulae.brew.sh/formula/imagemagick) to manipulate images into other formats
- [`stats`](https://github.com/exelban/stats) to see system stats
- [`Mole`](https://github.com/tw93/Mole) to see system stats and cleanup

## Code

`brew install --cask`:

- [R](https://formulae.brew.sh/cask/r)
	- I would recommend only installing `renv` into your base distribution, and then using `renv` for virtual environments for each individual project. This creates a `renv.lock` file which tells users exactly which version of each package you used for the analysis, making the research more reproducible.
	- Most people use RStudio, but I use VSCode. For use with VSCode and the [VScode-R](https://marketplace.visualstudio.com/items?itemName=REditorSupport.r) extension, you will also need to install `languageserver`, `jsonlite`, `rlang`. For an enhanced experience, install [radian, edit the Rterm path and enable bracketed paste](https://www.r-bloggers.com/2021/01/setup-visual-studio-code-to-run-r-on-vscode-2021/) (and maybe add an argument to [`r.rterm.option`](https://github.com/randy3k/radian/issues/372#issuecomment-1288615183))
	- I created a [cookiecutter](https://github.com/sparklabnyc/cookiecutter-r-project)template for R analysis projects.
- [Quarto](https://quarto.org/) for technical documents
- python ([[./python versioning, virtual environments and packaging|uv]] for python versioning and virtual environments)
	- Formerly used [pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#set-up-your-shell-environment-for-pyenv), venv, pip but uv replaces all of these
	- Former conda/mamba/micromamba user, trying out the recommended stuff before probably switching to [pixi](https://github.com/prefix-dev/pixi). Further notes on [[./python versioning, virtual environments and packaging|uv]].
- obsidian
	- daily notes in `notes/daily` with template `templates/daily`. template folder `templates/`
	- community plugins
		- advanced tables
		- auto note mover (exclude `notes/daily`)
		- clear unused images
		- code block enhancer (ignore `todoist`)
		- dataview
		- git
		- github publisher (with [[./getting these notes online|quartz and github pages for site]])
		- homepage
		- linter
		- minimal theme
		- paste URL into section
		- quick latex
		- rollover daily todos
		- tasks
		- templater
	- [matcha](https://github.com/piqoni/matcha) to pull my rss feeds, all situated in a folder `feed/`

## Applications

`brew install --cask` :

- [VSCode](https://formulae.brew.sh/cask/visual-studio-code) IDE. I use VSCode for everything because I can have one environment for all programming languages.
	- Citation Picker for Zotero, Code Spell Checker, Excel Viewer, GitHub Copilot, Markdown All in One, markdownlint, python, quarto, R, Rainbow CSV, White theme, stan-vscode, vscode-pdf, gitlens, autodocstring, TODOtree, ruff
	- Settings. Terminal › External: Osx Exec: iTerm.app
- [bitwarden](https://formulae.brew.sh/cask/bitwarden)password manager.
- [Zotero](https://formulae.brew.sh/cask/zotero) reference management.
	- [Better BibTex](https://retorque.re/zotero-better-bibtex/) extension
- [Arc](https://formulae.brew.sh/cask/arc) browser. Trying this one – I used Brave before.
	- bitwarden extension
	- Zotero Connector extension
	- [Refined GitHub](https://github.com/refined-github/refined-github) extension
- Optional: docker, zoom, microsoft office

## Themes and colours

Mostly inspired by the [Hundred Rabbits](https://100r.co) artist collective and their site. [My website](https://theorashid.github.io/) is also inspired by them (see [style.css](https://github.com/theorashid/theorashid.github.io/blob/master/style.css#L65-L66)).

- oh-my-zsh. `ZSH_THEME="minimal"`
- VSCode. [White](https://marketplace.visualstudio.com/items?itemName=arthurwhite.white) theme.
- Obsidian. Minimal theme.
Monochrome palette:
- `["#FBFBFB", "#222222", "#777777", "#727272"]`

See also [*How to Set up an Apple Mac for Software Development*](https://www.stuartellis.name/articles/mac-setup/).
