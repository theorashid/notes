---
tags:
  - website
folder: learning
share: true
title: getting these notes online
date created: Sunday, February 25th 2024, 8:53:49 pm
date modified: Sunday, February 25th 2024, 9:53:09 pm
---

These online [notes](https://theorashid.github.io/notes) are originally written in my Obsidian vault. They are styled using the static website generator [quartz](https://quartz.jzhao.xyz/), which has a number of great plugins to get the most out of notes made in Obsidian specifically, such as graph view and wikilinks.

I followed the [quartz documentation](https://quartz.jzhao.xyz/) and Nicole van der Hoeven's [video](https://www.youtube.com/watch?v=6s6DT1yN4dw) to set up the website and publish to GitHub pages, but found there were two issues I needed to fix:

1. I don't want *all* my Obsidian notes online (**no personal** stuff) and I didn't want to separate my single vault into a personal and public vault as a workaround.
2. quartz doesn't support **dataview queries**.

Both of these issues were solved by the [Obsidian GitHub Publisher](https://github.com/ObsidianPublisher/obsidian-github-publisher) plugin.

## removing personal notes

Obsidian GitHub Publisher takes the markdown notes in your vault and pushes them to a remote GitHub repository. I configured the plugin to populate the `content/` folder of the *same* repository as the quartz site with a [fixed folder file tree](https://obsidian-publisher.netlify.app/plugin/settings/upload/#fixed-folder) with the root folder `content/`. This means all files end up in `content/` root regardless of the Obsidian file structure.

I added the folders `templates`, `daily`, and other personal folders to the excluded folders zone. I also configured a share key, such that only notes with `share: true` in the frontmatter were pushed to GitHub. quartz has [its own method of hiding private files](https://quartz.jzhao.xyz/features/private-pages) with `draft: false` in the frontmatter. I didn't want two new properties in the frontmatter, so I edited the `draft.ts` file in quartz to filter on `share` instead.

```ts
import { QuartzFilterPlugin } from "../types"

export const RemoveDrafts: QuartzFilterPlugin<{}> = () => ({
  name: "RemoveDrafts",
  shouldPublish(_ctx, [_tree, vfile]) {
    const draftFlag: boolean = vfile.data?.frontmatter?.share ?? true
    return draftFlag
  },
})
```

## supporting dataview queries

Obsidian GitHub Publisher has the option to convert dataview to markdown before pushing to the remote.

## other tweaks

- Removed any files relating to dark mode and simplified the colour palettes in `quartz.config.ts`.
- Simplified the footer to link back to the main site only.
- Turned off the article read time.
- Customised the css to match my [site](https://theorashid.github.io/notes/).
