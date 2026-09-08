"""Compute the complete EGMS-QA task system in dependency order.

Inputs must be installed or prepared locally. New tiles use the released
training population as their frozen reference; no target rows enter fitting.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from graphlib import TopologicalSorter
import hashlib
from importlib.metadata import version
from itertools import chain
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from egms_qa.paths import DATA_DIR, ENCODER_TOKENS, OUTPUTS_DIR, SPLIT_MANIFEST, TASKS_DIR

GROUPS = tuple(f"{family}{i}" for family, count in (("a", 5), ("b", 6), ("c", 5), ("d", 4), ("s", 4), ("x", 3)) for i in range(1, count + 1))
DEPENDENCIES = {g: () for g in GROUPS}
DEPENDENCIES.update(a5=("a1", "a2", "a3", "a4"), b6=("b3", "b4"),
                    c5=("b2", "b3", "b6", "c3"), d2=("b5",),
                    d4=("b3", "b4", "b5", "d1", "d2", "d3"),
                    s3=("a4", "b3", "b4", "b5", "c1", "c2", "c3", "c4", "d1", "d2", "d3", "s2"))
CALIBRATED = {"a1", "a2", "c1", "c2", "c3", "d1", "d4", "s1", "s2", "s3", "s4"}
MEASUREMENTS = {"a1", "a2", "a4", "b1", "b2", "b3", "b4", "b5", "c1", "c2", "c3", "c4", "d1", "d2", "d3"}
WORKER_GROUPS = {"b1", "b2", "b3", "b4", "b5", "c1", "c2", "c3", "c4", "d1", "d2", "d3"}
SCHEMA = "egms-qa-task-run-1"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("reproduce", "new-tiles"), default="reproduce")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--data-config", type=Path, default=DATA_DIR / "encoder/manifest/data_config.json")
    p.add_argument("--source-tiles-root", type=Path)
    p.add_argument("--encoder-dir", type=Path, default=DATA_DIR / "encoder/checkpoint")
    p.add_argument("--token-cache", type=Path, help="New tiles are encoded automatically when omitted in new-tiles mode.")
    p.add_argument("--reference-tables", type=Path, default=TASKS_DIR)
    p.add_argument("--reference-token-cache", type=Path, default=ENCODER_TOKENS)
    p.add_argument("--out-dir", type=Path, default=OUTPUTS_DIR / "tasks-rebuilt")
    p.add_argument("--device", choices=("auto", "cpu", "cuda:0"), default="auto")
    p.add_argument("--workers", type=int, default=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p


def task_order():
    return tuple(TopologicalSorter(DEPENDENCIES).static_order())


def table_path(root, group):
    return root / group / f"{group}_final_table.csv"


def command(module, **options):
    result = [sys.executable, "-m", module]
    for name, value in options.items():
        if value is not None:
            result.extend(["--" + name.replace("_", "-"), str(value)])
    return result


def build_plan(args, n_tiles):
    root = args.out_dir
    manifest = root / "inputs/manifest.parquet"
    tokens = args.token_cache or (ENCODER_TOKENS.resolve() if args.mode == "reproduce" else root / "tokens/egms_tokens.pt")
    reference = root / "reference_state.joblib" if args.mode == "new-tiles" else None
    plans = []
    for group in task_order():
        options = {}
        if group in MEASUREMENTS:
            options["manifest"] = manifest
        if group in {"a1", "a2", "a4", "d1", "d2"}:
            options["data_config"] = args.data_config
        if group in {"a1", "a3", "s1", "s2", "s4"}:
            options["token_cache"] = tokens
        if group in {"a1", "a2"}:
            options.update(checkpoint=args.encoder_dir / "encoder.safetensors",
                           model_config=args.encoder_dir / "config.json", normalization=args.encoder_dir / "normalization.json",
                           sample_tiles=n_tiles, num_shards=1, shard_index=0, device=args.device,
                           out_dir=root / group / "work/shards/shard_0")
            if group == "a2":
                options["training_args"] = args.encoder_dir / "training_args.json"
        elif group in {"a3", "a4", "a5"}:
            options["out_path"] = table_path(root, group)
        else:
            options["out_dir"] = root / group
        if group in WORKER_GROUPS:
            options["workers"] = args.workers
        if group == "a5":
            options.update({f"{g}_path": table_path(root, g) for g in DEPENDENCIES[group]})
        if group in {"b6", "d4"}:
            options.update({f"{g}_table": table_path(root, g) for g in DEPENDENCIES[group]})
        if group == "c5":
            options.update({g: table_path(root, g) for g in DEPENDENCIES[group]})
        if group == "d2":
            options["b51_table"] = table_path(root, "b5")
        if group == "s2":
            options["n_jobs"] = args.workers
        if group == "s3":
            options["tasks_root"] = root
        if group in CALIBRATED - {"a1", "a2"}:
            options["reference_state"] = reference
        commands = [command(f"egms_qa.qa_construction.tasks.{group}.{group}_compute", **options)]
        if group == "c4":
            commands[0].insert(3, "final")
        if group in {"a1", "a2"}:
            commands.append(command(f"egms_qa.qa_construction.tasks.{group}.{group}_combine_shards",
                                    base_dir=root / group / "work", num_shards=1,
                                    out_path=table_path(root, group), reference_state=reference))
        plans.append({"group": group, "dependencies": list(DEPENDENCIES[group]), "commands": commands,
                      "table": str(table_path(root, group))})
    extraction = None
    if args.mode == "new-tiles" and args.token_cache is None:
        extraction = command("egms_encoder.extract_tokens", checkpoint=args.encoder_dir / "encoder.safetensors",
                             model_config=args.encoder_dir / "config.json", normalization=args.encoder_dir / "normalization.json",
                             manifest=manifest, data_config=args.data_config, output_dir=root / "tokens",
                             output_name="egms_tokens.pt", device=args.device)
    return {"mode": args.mode, "tiles": n_tiles, "tokens": str(tokens), "reference": str(reference) if reference else None,
            "extract_tokens": extraction, "tasks": plans}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def fingerprints(paths, workers=4):
    names = sorted({str(Path(p).resolve()) for p in paths})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(zip(names, pool.map(sha256, names)))


def save_state(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def validate_tokens(path, frame, encoder_hashes):
    import torch
    cache = torch.load(path, map_location="cpu", weights_only=True)
    n = len(frame)
    if tuple(cache["spatial_tokens"].shape) != (n, 65, 256) or tuple(cache["token_mask"].shape) != (n, 65):
        raise ValueError("token cache must match the manifest and have [N,65,256] features and [N,65] masks")
    if cache["token_mask"].dtype != torch.bool or tuple(cache["point_count_per_bin"].shape) != (n, 64):
        raise ValueError("invalid token validity mask or point-count layout")
    ids = [str(t) for t in cache["tile_ids"]]
    if len(ids) != n or len(set(ids)) != n or len(cache["splits"]) != n:
        raise ValueError("token cache has duplicate IDs or inconsistent lengths")
    actual = dict(zip(ids, cache["splits"]))
    if actual != dict(zip(frame["tile_id"].astype(str), frame["split"].astype(str))):
        raise ValueError("token cache tile IDs or splits do not match the manifest")
    metadata = cache.get("metadata", {}).get("reproduction_input_sha256", {})
    if any(metadata.get(k) != v for k, v in encoder_hashes.items()):
        raise ValueError("token cache does not identify the selected encoder and normalization; regenerate tokens with egms_encoder.extract_tokens")


def validate_table(path, frame, group):
    import pandas as pd
    from .task_specs import TASK_SPECS
    table = pd.read_csv(path)
    if group.startswith("x"):
        expected = {"x1": 5, "x2": 6, "x3": 3}[group]
        if not {"task_id", "target_column"} <= set(table) or len(table) != expected or table["task_id"].duplicated().any():
            raise ValueError(f"invalid {group} refusal catalog")
        if set(table["task_id"]) != {f"{group.upper()}{i}" for i in range(1, expected + 1)}:
            raise ValueError(f"unexpected task IDs in {group}")
    else:
        needed = {s["target_column"] for s in TASK_SPECS if s["source_family"] == group} | {"tile_id", "split"}
        if not needed <= set(table) or table["tile_id"].duplicated().any():
            raise ValueError(f"missing columns or duplicate IDs in {group}")
        actual = set(zip(table["tile_id"].astype(str), table["split"].astype(str)))
        if len(table) != len(frame) or actual != set(zip(frame["tile_id"].astype(str), frame["split"].astype(str))):
            raise ValueError(f"{group} does not contain exactly the requested tile/split keys")


def execute(commands, log):
    with log.open("a") as stream:
        for cmd in commands:
            stream.write(json.dumps(cmd) + "\n")
            stream.flush()
            subprocess.run(cmd, check=True, stdout=stream, stderr=subprocess.STDOUT)


def check_output_root(root, inputs, resume):
    if root.is_symlink() or root == TASKS_DIR.resolve():
        raise ValueError("output directory must not be the installed reference directory or a symlink")
    if any(root == Path(p).resolve() or root in Path(p).resolve().parents for p in inputs):
        raise ValueError("output directory must not contain input files")
    if root.exists() and not resume:
        raise FileExistsError("output directory already exists; choose a new directory or use --resume")
    if resume and not (root / "run.json").is_file():
        raise ValueError("--resume requires a previous run.json")


def prepare_manifest(frame, data_config, workers):
    """Derive optional manifest metadata and validate the complete tile contract."""
    import numpy as np
    from egms_encoder.data.tile_store import TileStore, TimeWindow
    frame = frame.copy()
    missing_metadata = not {"n_points", "centroid_x", "centroid_y"} <= set(frame)
    if missing_metadata:
        def details(path):
            with np.load(path, allow_pickle=False) as z:
                coords = z["coords"]
                if coords.ndim != 2 or coords.shape[1] != 2 or not len(coords):
                    raise ValueError(f"invalid coordinates in {path}")
                center = coords.mean(axis=0)
                return len(coords), float(center[0]), float(center[1])
        with ThreadPoolExecutor(max_workers=workers) as pool:
            values = list(pool.map(details, frame["path"]))
        for i, key in enumerate(("n_points", "centroid_x", "centroid_y")):
            frame[key] = [v[i] for v in values]
    config = json.loads(data_config.read_text())
    store = TileStore(frame, TimeWindow.from_config(config),
                      split_assignments=dict(zip(frame["tile_id"], frame["split"])), data_config=config)
    def validate(index):
        store.get_tile(index)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(validate, range(len(frame))))
    return frame


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.workers < 1:
        p.error("--workers must be positive")
    if args.mode == "new-tiles" and args.manifest is None:
        p.error("--manifest is required for new tiles")
    args.manifest = args.manifest or SPLIT_MANIFEST
    if args.out_dir.is_symlink():
        p.error("output directory must not be a symlink")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    from .inputs import read_tile_manifest
    frame = read_tile_manifest(args.manifest, args.source_tiles_root)
    if args.mode == "reproduce" and frame["split"].value_counts().to_dict() != {"train": 8000, "val": 1000, "test": 1000}:
        p.error("reproduce mode requires the released 8,000/1,000/1,000 split; use --mode new-tiles for another collection")
    if args.dry_run:
        print(json.dumps(build_plan(args, len(frame)), indent=2))
        return
    import torch
    from .temporal_inputs import TimeAxis
    TimeAxis.from_file(args.data_config)
    if args.device == "auto":
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    encoder_files = {"encoder_weights": args.encoder_dir / "encoder.safetensors",
                     "encoder_config": args.encoder_dir / "config.json", "normalization": args.encoder_dir / "normalization.json"}
    input_files = [args.manifest, args.data_config, *encoder_files.values(), args.encoder_dir / "training_args.json", *frame["path"]]
    plan = build_plan(args, len(frame))
    if not plan["extract_tokens"]:
        input_files.append(plan["tokens"])
    if args.mode == "new-tiles":
        input_files += [args.reference_token_cache, *[table_path(args.reference_tables, g) for g in GROUPS if not g.startswith("x")]]
    check_output_root(args.out_dir, input_files, args.resume)
    print(f"Checking inputs for {len(frame)} tiles on {args.device}", flush=True)
    import egms_encoder
    code_files = chain(Path(__file__).resolve().parents[1].rglob("*.py"),
                       Path(egms_encoder.__file__).resolve().parent.rglob("*.py"))
    signature = {"options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in {"resume", "dry_run"}},
                 "inputs": fingerprints(input_files, args.workers), "code": fingerprints(code_files, args.workers),
                 "environment": {"python": sys.version.split()[0], **{name: version(name) for name in
                                 ("numpy", "pandas", "torch", "scipy", "scikit-learn", "joblib", "safetensors")}}}
    encoder_hashes = {k: signature["inputs"][str(v.resolve())] for k, v in encoder_files.items()}
    target_hashes = {**encoder_hashes, "data_config": signature["inputs"][str(args.data_config)]}
    state_path = args.out_dir / "run.json"
    if args.resume:
        state = json.loads(state_path.read_text())
        if state.get("schema") != SCHEMA or state["signature"] != signature:
            raise ValueError("run inputs, parameters or code changed; start a new output directory")
        for name, digest in state.get("prepared", {}).items():
            if not Path(name).is_file() or sha256(name) != digest:
                raise ValueError(f"prepared input changed: {name}")
    else:
        frame = prepare_manifest(frame, args.data_config, args.workers)
        args.out_dir.mkdir(parents=True, exist_ok=False)
        state = {"schema": SCHEMA, "signature": signature, "plan": plan, "prepared": {}, "groups": {}, "status": "running"}
        (args.out_dir / "inputs").mkdir()
        frame.to_parquet(args.out_dir / "inputs/manifest.parquet", index=False)
        (args.out_dir / "logs").mkdir()
        save_state(state_path, state)
    try:
        if not state.get("preparation_complete"):
            if plan["extract_tokens"]:
                print("Extracting tokens for new tiles", flush=True)
                execute([plan["extract_tokens"]], args.out_dir / "logs/tokens.log")
            validate_tokens(plan["tokens"], frame, target_hashes)
            if plan["reference"]:
                print("Building the frozen release reference", flush=True)
                from .reference import build_reference
                ref = torch.load(args.reference_token_cache, map_location="cpu", weights_only=True)
                import pandas as pd
                ref_frame = pd.DataFrame({"tile_id": ref["tile_ids"], "split": ref["splits"]})
                validate_tokens(args.reference_token_cache, ref_frame, encoder_hashes)
                del ref
                build_reference(args.reference_tables, args.reference_token_cache, Path(plan["reference"]))
            prepared = [args.out_dir / "inputs/manifest.parquet"]
            if plan["extract_tokens"]:
                prepared.append(plan["tokens"])
            if plan["reference"]:
                prepared.append(plan["reference"])
            state["prepared"] = fingerprints(prepared, args.workers)
            state["preparation_complete"] = True
            save_state(state_path, state)
        for item in plan["tasks"]:
            group = item["group"]
            previous = state["groups"].get(group)
            directory = args.out_dir / group
            if directory.is_symlink():
                raise ValueError(f"task output is a symlink: {directory}")
            if previous and previous["status"] == "complete":
                current = fingerprints(directory.rglob("*.*"), args.workers)
                if current != previous["artifacts"]:
                    raise ValueError(f"completed output changed for {group}; start a new run")
                print(f"{group}: verified, skipping", flush=True)
                continue
            if directory.exists():
                if previous is None:
                    raise FileExistsError(f"unexpected task directory: {directory}")
                shutil.rmtree(directory)
            state["groups"][group] = {"status": "running"}
            save_state(state_path, state)
            print(f"{group}: running", flush=True)
            execute(item["commands"], args.out_dir / "logs" / f"{group}.log")
            validate_table(Path(item["table"]), frame, group)
            state["groups"][group] = {"status": "complete", "artifacts": fingerprints(directory.rglob("*.*"), args.workers)}
            save_state(state_path, state)
        state["status"] = "complete"
        state.pop("error", None)
        save_state(state_path, state)
        print(f"Completed all {len(GROUPS)} task groups: {args.out_dir}", flush=True)
    except Exception as error:
        state["status"] = "failed"
        state["error"] = str(error)
        save_state(state_path, state)
        raise


if __name__ == "__main__":
    main()
