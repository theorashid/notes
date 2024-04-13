---
tags:
  - python
  - swe
folder: learning
share: true
title: Typer to replace argparse
date created: Saturday, April 13th 2024, 11:42:55 am
date modified: Saturday, April 13th 2024, 11:57:32 am
---

Sometimes in scripts, it is hard to keep track of the number and function of the arguments when using `argparse`.

```python
import argparse

def read_data(args):
	# difficult to keep track of `args`
	df_club = pd.read_csv(args.club_fname)
	df_player = pd.read_csv(args.player_fname)

def main(args):
	read_data(args)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--club-fname", type=str)
	parser.add_argument("--player-fname", type=str)
	args = parser.parse_args()

	main(args)
```

We can use [Typer](https://typer.tiangolo.com/) to build a CLI which keeps track of the name of the arguments.

```python
import typer

app = typer.Typer()

def read_data(club_fname: str, player_fname: str):
	df_club = pd.read_csv(club_fname)
	df_player = pd.read_csv(player_fname)

@app.command()
def main(
	club_fname: Annotated[str, typer.Option(help="path to club data")],
	player_fname: Annotated[str, typer.Option()],
):
	read_data(club_fname, player_fname)

if __name__ == "__main__":
	app()
```

It also provides a nice interface with type hints if we run `python script.py --help`.
