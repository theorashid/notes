---
tags:
  - visualisation
  - swe
folder: learning
share: true
title: observable plot
date created: Wednesday, October 2nd 2024, 4:42:31 pm
date modified: Wednesday, October 2nd 2024, 5:09:16 pm
---

**[Observable Plot](https://observablehq.com/plot/features/plots)** is a high-level JavaScript plotting library built on top of (the low-level) [[./d3.js|d3.js]]. Uses the *grammar of graphics* style like [`ggplot2`](https://ggplot2.tidyverse.org/).

```js
Plot.plot({
  // plot sizing
  height: 800,
  marginRight: 90,
  marginLeft: 110,
  x: { ticks: 0 },  // x-axis without labels
  y: {
    nice: true,
    grid: true,  // grid-lines
    label: "y-label"
  },
  color: {type: "categorical"},  // colourscheme for stroke
  marks: [
    Plot.frame(),  // box around the plot
    Plot.dot(
      data,
      { 
        x: "variable",
        y: "outcome_variable",
        fy: "variable_to_facet",  // facet y-direction
        stroke: "categorical_variable",
      }
    )
  ]
})
```
