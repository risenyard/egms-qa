# X2 Algorithm: Unsupported Data And Scope Boundary

## Delivered Tasks

- `X21_exact_asset_refusal`
- `X22_point_subcell_exact_refusal`
- `X23_unsupported_displacement_component_refusal`
- `X24_external_context_refusal`
- `X25_live_status_refusal`
- `X26_unsupported_ranking_superlative_refusal`

## Role

X2 is a boundary/refusal family for questions that exceed the available input
channels, spatial scale, temporal status, or reference universe.

## Rule

When a question asks for exact assets, point/sub-cell conclusions, unsupported
displacement components, external imagery/land-use/geology context, live status,
or open-world rankings/superlatives, return a refusal-style answer:

```text
1. State that the requested scope is not supported.
2. Name the missing data channel, scale, time status, or ranking universe.
3. Redirect to supported tile-level, bin-level, or corpus-relative tasks.
```

Europe-defined and corpus-relative classes are allowed when the task explicitly
defines that reference distribution. X26 only rejects undefined rank, worst,
most severe, or highest-risk claims.

## Answer Guidance

Every X2 catalog row includes:

- `answer_policy`: the family-level refusal policy.
- `response_template`: a task-specific answer template suitable for VQA
  generation.
- `supported_redirect_tasks`: concrete A/B/C/D/S task IDs that can answer the
  nearest supported tile-level, bin-level, or corpus-relative question.

Generic pattern:

```text
Cannot answer the requested scale, data channel, live status, or undefined
ranking from the current inputs. State the missing scale/channel/time/reference
universe, then redirect to supported tile-level, 8x8 bin-level, or predefined
Europe/corpus-relative tasks.
```

## File Inventory

- `x2_final_table.csv`: static boundary catalog, one row per X2 task, including answer guidance and redirect task IDs.
- `x2_compute.py`: reproducible catalog generator.
