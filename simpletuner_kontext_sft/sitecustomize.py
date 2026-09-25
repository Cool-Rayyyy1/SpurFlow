"""Skip SimpleTuner NFS directory walks that freeze Kontext SFT startup.

SimpleTuner's LocalDataBackend.list_files() rglobs and follows every symlink.
pico-banana train/edit has ~250k image links; that takes 20-40 minutes per
dataset with GPU util 0. Read bucket metadata / all_image_files JSON instead.
Also skip fcntl.flock on NFS (can fail and force a relist).
"""

from __future__ import annotations

import importlib.machinery
import json
import os
import sys
from pathlib import Path


_PATCHED = set()
_TARGETS = {
    "simpletuner.helpers.data_backend.local",
    "simpletuner.helpers.training.state_tracker",
    "simpletuner.helpers.caching.vae",
    "simpletuner.helpers.data_backend.factory",
}


def _log(msg: str) -> None:
    print(f"[simpletuner-patch] {msg}", flush=True)


_META_NAMES = (
    "aspect_ratio_bucket_metadata_pico-banana-edit.json",
    "aspect_ratio_bucket_metadata_pico-banana-reference.json",
    "aspect_ratio_bucket_metadata_pico-banana-edit-val.json",
    "aspect_ratio_bucket_metadata_pico-banana-reference-val.json",
    "aspect_ratio_bucket_metadata_oss-edit.json",
    "aspect_ratio_bucket_metadata_oss-reference.json",
)

_FILELIST_NAMES = (
    "all_image_files_pico-banana-edit.json",
    "all_image_files_pico-banana-reference.json",
    "all_image_files_pico-banana-edit-val.json",
    "all_image_files_pico-banana-reference-val.json",
    "all_image_files_oss-edit.json",
    "all_image_files_oss-reference.json",
)


def _filter_paths(paths, extensions: list[str] | None) -> list[str]:
    exts = {f".{e.lstrip('.').lower()}" for e in (extensions or []) if e}
    if not exts:
        return list(paths)
    return [p for p in paths if Path(p).suffix.lower() in exts]


def _backend_meta_name(instance_data_dir: str) -> str | None:
    path = instance_data_dir.replace("\\", "/")
    parts = Path(instance_data_dir).parts
    leaf = parts[-1] if parts else ""
    split = parts[-2] if len(parts) >= 2 else ""
    if "pico-banana" in path:
        if split == "val":
            return "pico-banana-edit-val" if leaf == "edit" else "pico-banana-reference-val"
        return "pico-banana-edit" if leaf == "edit" else "pico-banana-reference"
    if "oss_edit" in path or "/oss/" in path:
        return "oss-edit" if leaf == "edit" else "oss-reference"
    return None


def _image_files_from_json(instance_data_dir: str, extensions: list[str] | None) -> list[str]:
    folder = Path(instance_data_dir)
    # Exact filenames only. glob/listdir on the pair dir is the 20-40min NFS hang.
    names = []
    bid = _backend_meta_name(instance_data_dir)
    if bid:
        names.append(f"aspect_ratio_bucket_metadata_{bid}.json")
    names.extend(_META_NAMES)
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        meta_path = folder / name
        try:
            if not meta_path.is_file():
                continue
        except OSError:
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as exc:
            _log(f"failed to read {meta_path.name}: {exc}")
            continue
        if not isinstance(meta, dict) or not meta:
            continue
        files = _filter_paths(meta.keys(), extensions)
        if files:
            _log(f"list_files {folder} <- {meta_path.name} n={len(files)}")
            return files

    output_dir = os.environ.get("SIMPLETUNER_OUTPUT_DIR", "")
    if output_dir:
        out = Path(output_dir)
        filelist_names = []
        if bid:
            filelist_names.append(f"all_image_files_{bid}.json")
        filelist_names.extend(_FILELIST_NAMES)
        seen = set()
        prefix = str(folder)
        for name in filelist_names:
            if name in seen:
                continue
            seen.add(name)
            json_path = out / name
            if not json_path.is_file():
                continue
            try:
                payload = json.loads(json_path.read_text())
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            matched = _filter_paths(
                (p for p in payload.keys() if str(p).startswith(prefix)),
                extensions,
            )
            if matched:
                _log(f"list_files {folder} <- {json_path.name} n={len(matched)}")
                return matched
    return []


def _is_huge_pair_dir(instance_data_dir: str) -> bool:
    path = instance_data_dir.replace("\\", "/")
    return "simpletuner_pairs" in path


def _patch_local(module) -> None:
    cls = getattr(module, "LocalDataBackend", None)
    if cls is None or getattr(cls.list_files, "_editflow_patched", False):
        return

    orig = cls.list_files

    def list_files(self, file_extensions, instance_data_dir):
        if not instance_data_dir:
            return orig(self, file_extensions, instance_data_dir)
        # Only the pico/oss pair dirs are huge. Text/VAE cache dirs are tiny and
        # must be allowed to list (they have no metadata JSON).
        if not _is_huge_pair_dir(instance_data_dir):
            return orig(self, file_extensions, instance_data_dir)
        files = _image_files_from_json(instance_data_dir, file_extensions)
        if files:
            return [(os.path.abspath(instance_data_dir), [], files)]
        raise FileNotFoundError(
            f"Refusing to listdir {instance_data_dir}: no aspect_ratio_bucket_metadata_*.json "
            "or all_image_files_*.json. Scanning this folder on NFS takes 20-40 minutes."
        )

    list_files._editflow_patched = True
    cls.list_files = list_files
    _log("patched LocalDataBackend.list_files")


def _patch_state_tracker(module) -> None:
    cls = getattr(module, "StateTracker", None)
    if cls is None:
        return
    if getattr(cls._load_from_disk, "_editflow_patched", False):
        return

    @classmethod
    def _load_from_disk(cls, cache_name, retry_limit: int = 0):
        cache_path = Path(cls.args.output_dir) / f"{cache_name}.json"
        if not cache_path.exists():
            return None
        try:
            with cache_path.open("r") as handle:
                return json.load(handle)
        except Exception as exc:
            _log(f"cache load failed {cache_path.name}: {exc}")
            return None

    @classmethod
    def _save_to_disk(cls, cache_name, data):
        cache_path = Path(cls.args.output_dir) / f"{cache_name}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".json.tmp")
        with tmp.open("w") as handle:
            json.dump(data, handle)
        tmp.replace(cache_path)

    _load_from_disk._editflow_patched = True
    cls._load_from_disk = _load_from_disk
    cls._save_to_disk = _save_to_disk
    _log("patched StateTracker cache json I/O (no flock)")


def _patch_vae(module) -> None:
    cls = getattr(module, "VAECache", None)
    if cls is None:
        return
    if getattr(cls.build_vae_cache_filename_map, "_editflow_patched", False):
        return

    orig_build = cls.build_vae_cache_filename_map
    orig_cached = cls.already_cached

    def build_vae_cache_filename_map(self, all_image_files):
        self.image_path_to_vae_path = {}
        self.vae_path_to_image_path = {}
        if self.vae_cache_ondemand or self.vae_cache_disable:
            _log(f"skip VAE filename map id={self.id} (ondemand, lazy paths)")
            return
        return orig_build(self, all_image_files)

    def already_cached(self, filepath: str) -> bool:
        mapping = getattr(self, "image_path_to_vae_path", None)
        if mapping is None:
            self.image_path_to_vae_path = {}
            self.vae_path_to_image_path = {}
            mapping = self.image_path_to_vae_path
        test_path = mapping.get(filepath)
        if not test_path:
            test_path, _ = self.generate_vae_cache_filename(filepath)
            if self.cache_data_backend.type == "local":
                test_path = os.path.abspath(test_path)
            mapping[filepath] = test_path
            self.vae_path_to_image_path[test_path] = filepath
        return bool(test_path) and self.cache_data_backend.exists(test_path)

    build_vae_cache_filename_map._editflow_patched = True
    already_cached._editflow_patched = True
    cls.build_vae_cache_filename_map = build_vae_cache_filename_map
    cls.already_cached = already_cached
    _log("patched VAECache: ondemand skips full file list")


def _patch_factory(module) -> None:
    cls = getattr(module, "FactoryRegistry", None)
    if cls is None or getattr(cls._configure_vae_cache, "_editflow_patched", False):
        return

    orig = cls._configure_vae_cache

    def _configure_vae_cache(
        self,
        backend,
        init_backend,
        image_embed_data_backend,
        vae_cache_dir_paths,
        text_embed_cache_dir_paths,
        conditioning_type,
    ):
        skip_list = bool(
            getattr(self.args, "vae_cache_ondemand", False) or getattr(self.args, "vae_cache_disable", False)
        )
        if not skip_list:
            return orig(
                self,
                backend,
                init_backend,
                image_embed_data_backend,
                vae_cache_dir_paths,
                text_embed_cache_dir_paths,
                conditioning_type,
            )

        from simpletuner.helpers.training.state_tracker import StateTracker

        orig_get = StateTracker.get_image_files.__func__

        def _get(cls_, data_backend_id, retry_limit=0):
            result = orig_get(cls_, data_backend_id, retry_limit=0)
            if result is None:
                _log(f"skip pair-dir list_files for {data_backend_id} (VAE ondemand)")
                return {}
            return result

        StateTracker.get_image_files = classmethod(_get)
        try:
            return orig(
                self,
                backend,
                init_backend,
                image_embed_data_backend,
                vae_cache_dir_paths,
                text_embed_cache_dir_paths,
                conditioning_type,
            )
        finally:
            StateTracker.get_image_files = classmethod(orig_get)

    _configure_vae_cache._editflow_patched = True
    cls._configure_vae_cache = _configure_vae_cache
    _log("patched FactoryRegistry._configure_vae_cache")


class _PatchFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in _PATCHED or fullname not in _TARGETS:
            return None
        _PATCHED.add(fullname)
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            if fullname.endswith(".local"):
                _patch_local(module)
            elif fullname.endswith(".state_tracker"):
                _patch_state_tracker(module)
            elif fullname.endswith(".vae"):
                _patch_vae(module)
            elif fullname.endswith(".factory"):
                _patch_factory(module)

        spec.loader.exec_module = exec_module  # type: ignore[method-assign]
        return spec


sys.meta_path.insert(0, _PatchFinder())
_log("import hooks installed")
