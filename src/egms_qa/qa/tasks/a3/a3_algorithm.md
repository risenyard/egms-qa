# A31/A32 Algorithm

## Task

A31 measures spatial observation coverage:

> Are the observations spread across the tile, or concentrated into only a few spatial bins?

## Target

A31 uses the 8x8 spatial bin counts already stored in the EGMS encoder token cache:

```text
occupied_bins = count(point_count_per_bin > 0)
A31_valid_bin_fraction_8x8 = occupied_bins / 64
```

This is a raw observation-support target. It is not an encoder-advantage task.

## Classes

The class label uses fixed structural thresholds, not empirical percentiles:

| class | rule | meaning |
|---|---:|---|
| `well_spread` | fraction >= 0.75 | observations cover most of the tile |
| `moderate_gaps` | 0.50 <= fraction < 0.75 | visible coverage gaps |
| `sparse` | 0.25 <= fraction < 0.50 | sparse spatial support |
| `highly_fragmented` | fraction < 0.25 | very fragmented support |

Class counts in the final table:

| class | count | fraction |
|---|---:|---:|
| `well_spread` | 7836 | 0.7836 |
| `moderate_gaps` | 1663 | 0.1663 |
| `sparse` | 434 | 0.0434 |
| `highly_fragmented` | 67 | 0.0067 |

## Files

- `a3_final_table.csv`: final all10k table.
- `a3_compute.py`: deterministic computation from the EGMS encoder token cache.

The delivery folder intentionally does not retain exploration plots, scratch JSON
summaries, or intermediate analysis tables.
