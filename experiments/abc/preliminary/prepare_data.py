import argparse
import json
import os
import shutil
from pathlib import Path


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, mode: str) -> None:
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def copy_tree(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "copy":
        shutil.copytree(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst, target_is_directory=True)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def write_manifest(manifest_path: Path, manifest: dict) -> None:
    ensure_parent(manifest_path)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def parse_args():
    experiment_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_root",
        type=str,
        default="/group2/dgm/yonghyun/AttributeByUnlearning/custom_diffusion",
        help="Root of the AttributeByUnlearning custom_diffusion benchmark.",
    )
    parser.add_argument(
        "--data_dest",
        type=str,
        default=str(experiment_dir / "data"),
        help="Destination directory for abc data artifacts.",
    )
    parser.add_argument(
        "--config_dest",
        type=str,
        default=str(experiment_dir / "configs"),
        help="Destination directory for abc task metadata.",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "symlink"],
        default="symlink",
        help="Whether to copy the benchmark artifacts or symlink them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    source_data = source_root / "data" / "data" / "abc"
    source_tasks = source_root / "tasks" / "all_tasks.json"

    required_files = [
        source_data / "laion_latents.npy",
        source_data / "laion_text_embeddings.npy",
        source_data / "laion_subset",
        source_data / "json",
        source_tasks,
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required AttributeByUnlearning artifacts:\n" + "\n".join(missing)
        )

    data_dest = Path(args.data_dest)
    config_dest = Path(args.config_dest)

    copy_file(source_data / "laion_latents.npy", data_dest / "laion_latents.npy", args.mode)
    copy_file(
        source_data / "laion_text_embeddings.npy",
        data_dest / "laion_text_embeddings.npy",
        args.mode,
    )
    copy_tree(source_data / "laion_subset", data_dest / "laion_subset", args.mode)
    copy_tree(source_data / "json", data_dest / "json", args.mode)
    copy_file(source_tasks, config_dest / "all_tasks.json", args.mode)

    with open(source_tasks, "r") as f:
        tasks = json.load(f)

    relative_assets = set()
    for task in tasks:
        for key in ("model_path", "synth_image_path"):
            rel_path = task.get(key)
            if rel_path:
                relative_assets.add(rel_path)
                if key == "model_path":
                    relative_assets.add(rel_path.replace("models", "models-ti"))
                if key == "synth_image_path":
                    relative_assets.add(rel_path.replace("synth", "synth-ti"))

    for rel_path in sorted(relative_assets):
        src = source_data / rel_path
        dst = data_dest / rel_path
        if src.is_dir():
            copy_tree(src, dst, args.mode)
        elif src.exists():
            copy_file(src, dst, args.mode)

    manifest = {
        "source_root": str(source_root),
        "mode": args.mode,
        "artifacts": {
            "laion_latents": str(data_dest / "laion_latents.npy"),
            "laion_text_embeddings": str(data_dest / "laion_text_embeddings.npy"),
            "laion_subset": str(data_dest / "laion_subset"),
            "json": str(data_dest / "json"),
            "all_tasks": str(config_dest / "all_tasks.json"),
            "task_assets": sorted(relative_assets),
        },
    }
    write_manifest(config_dest / "data_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
