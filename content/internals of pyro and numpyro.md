---
tags:
  - bayesian
  - jax
  - python
folder: learning
share: true
title: internals of pyro and numpyro
date created: Friday, January 9th 2026, 5:35:49 pm
date modified: Friday, May 22nd 2026, 12:34:34 pm
---

[pyro](https://pyro.ai/examples/effect_handlers.html) and [numpyro](https://github.com/pyro-ppl/numpyro/blob/master/numpyro/handlers.py) are built using composable [effect handlers](https://num.pyro.ai/en/stable/tutorials/effect_handlers.html) for recording and modifying the internals of the probabilistic program.

The `Messenger` passes around messages (dictionaries) of the form (e.g. for `pyro.sample("x", dist.Bernoulli(0.5), infer={"enumerate": "parallel"}, obs=None)`)

```python
msg = {
    # The following fields contain the name, inputs, function, and output of a site.
    # These are generally the only fields you'll need to think about.
    "name": "x",
    "fn": dist.Bernoulli(0.5),
    "value": None,  # msg["value"] will eventually contain the value returned by pyro.sample
    "is_observed": False,  # because obs=None by default; only used by sample sites
    "args": (),  # positional arguments passed to "fn" when it is called; usually empty for sample sites
    "kwargs": {},  # keyword arguments passed to "fn" when it is called; usually empty for sample sites
    # This field typically contains metadata needed or stored by a particular inference algorithm
    "infer": {"enumerate": "parallel"},
    # The remaining fields are generally only used by Pyro's internals,
    # or for implementing more advanced effects beyond the scope of this tutorial
    "type": "sample",  # label used by Messenger._process_message to dispatch, in this case to _pyro_sample
    "done": False,
    "stop": False,
    "scale": torch.tensor(1.),  # Multiplicative scale factor that can be applied to each site's log_prob
    "mask": None,
    "continuation": None,
    "cond_indep_stack": (),  # Will contain metadata from each pyro.plate enclosing this sample site.
}
```

This message can be updated using effect handlers ([order matters](https://num.pyro.ai/en/stable/tutorials/effect_handlers.html#Composing-Handlers:-Nesting-and-Order) when composing - outer runs last). Some examples:

- [`condition`](https://github.com/pyro-ppl/numpyro/blob/master/numpyro/handlers.py#L414) on data, which changes the `is_observed` property to `True` and clamps an output `value` for sample (inference) statements
	- `substitute` sets the `value` but `is_observed` stays `False` (useful for substituting posterior samples for prediction)
- [`mask`](https://github.com/pyro-ppl/numpyro/blob/master/numpyro/handlers.py#L622) to set `"mask": True` so the sample statements are elementwise ignored from the log-probability calculation
- `scale` handler multiplies `msg["scale"]` by a positive factor, used internally for data subsampling (upweighting a mini-batch to represent the full dataset)
- `uncondition` handler forces observed sites to sample from their prior distribution instead of using the observed value
- `block` handler hides sites from outer handlers by setting `msg["stop"] = True`
- `lift` handler converts `numpyro.param` sites into `numpyro.sample` sites by providing a prior distribution
- `scope` handler prepends a prefix to all site names

Then, to calculate the log probability for a single site from within a `Messenger`, we do something like

```python
# _pyro_sample will be called once per pyro.sample site.
# It takes a dictionary msg containing the name, distribution,
# observation or sample value, and other metadata from the sample site.
def _pyro_sample(self, msg):
	# Any unobserved random variables will trigger this assertion.
	assert msg["name"] in self.data
	msg["value"] = self.data[msg["name"]]
	# Since we've observed a value for this site, we set the "is_observed" flag to True
	# This tells any other Messengers not to overwrite msg["value"] with a sample.
	msg["is_observed"] = True
	self.logp = self.logp + (msg["scale"] * msg["fn"].log_prob(msg["value"])).sum()
```

Looking a level of abstraction above, we can use `trace` on the conditioned model to record the inputs, distributions, and outputs of `sample` statements, and then calculate the log probability for all the sites.

```python
def make_log_joint(model):
    def _log_joint(cond_data, *args, **kwargs):
        with TraceMessenger() as tracer:
            with ConditionMessenger(data=cond_data):
                model(*args, **kwargs)

        trace = tracer.trace
        logp = 0.
        for name, node in trace.nodes.items():
            if node["type"] == "sample":
                if node["is_observed"]:
                    assert node["value"] is cond_data[name]
                logp = logp + node["fn"].log_prob(node["value"]).sum()
        return logp
    return _log_joint
```
