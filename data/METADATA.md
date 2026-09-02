# EGMS VLM Data

This workspace uses the no-tail EGMS L3 U parquet prepared by the original
EGMS encoder project.

Required local files:

```text
data/processed/egms_merged_U_no_tail.parquet
data/processed/metadata_no_tail.json
```

Current data shape:

```text
points: 2,196,525
time steps: 302
time range: 2019-01-07 to 2023-12-18
CRS: EPSG:3035
```

The removed tail date is `2023-12-24`, which was excluded before V3/V3.1
encoder and VLM experiments.

Large parquet files are local artifacts and are ignored by git.
