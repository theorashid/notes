---
tags:
  - python
folder: learning
share: true
title: xarray zarr s3
date created: Monday, March 10th 2025, 10:42:35 pm
date modified: Saturday, March 15th 2025, 9:31:56 pm
---

[Zarr](https://docs.xarray.dev/en/stable/user-guide/io.html#zarr) can store arrays in cloud-based object storage such as s3 as chunked, compressed, N-dimensional arrays.

Open a single file:

```python
path = f"s3://{bucket}/dataset.zarr"
s3_store = zarr.storage.FsspecStore.from_url(path)
ds = xr.open_zarr(store=s3_store, consolidated=True, chunks="auto")
```

Open [multiple files as a single dataset](https://docs.xarray.dev/en/stable/generated/xarray.open_mfdataset.html) in parallel (using dask) and combine along a dimension:

```python
s3 = s3fs.S3FileSystem()
zarr_files = s3.glob(f"{bucket}/*.zarr")

ds = xr.open_mfdataset(
	[f"s3://{file}" for file in zarr_files],
	engine="zarr",
	concat_dim="dim_to_concat_along",
	combine="nested",
	parallel=True,
)
```
