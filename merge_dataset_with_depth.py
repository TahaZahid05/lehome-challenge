
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import scripts.utils.dataset_processing as dp
import numpy as np

# Monkey-patch _fix_depth_data_format with a metadata-only pre-check
_original_fix = dp._fix_depth_data_format

def _safe_fix_depth(dataset_root: Path) -> None:
    data_root = Path(dataset_root).resolve() / "data"
    parquet_files = sorted(data_root.glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        return

    schema = pq.read_schema(parquet_files[0])
    if "observation.top_depth" not in schema.names:
        return

    idx = schema.get_field_index("observation.top_depth")
    field_type = str(schema.field(idx).type)

    # Skip only if already plain fixed_size_list (previously normalized)
    if "fixed_size_list" in field_type and "extension" not in field_type:
        print(f"[skip] {Path(dataset_root).name} — already normalized")
        return

    print(f"[fix] {Path(dataset_root).name} — normalizing {len(parquet_files)} file(s)...")

    for pf in parquet_files:
        try:
            table = pq.read_table(pf)
            if "observation.top_depth" not in table.column_names:
                continue

            depth_col = table["observation.top_depth"]

            # Memory-efficient: use numpy instead of to_pylist()
            depth_pd = depth_col.to_pandas()  # numpy arrays, not Python lists
            arr_3d = np.stack(depth_pd.values)  # shape: (N, H, W)
            N, H, W = arr_3d.shape
            dtype = pa.uint16() if arr_3d.dtype == np.uint16 else pa.float32()
            arr_3d = arr_3d.astype(np.uint16 if dtype == pa.uint16() else np.float32)

            # Build fixed_size_list without Python object explosion
            flat = pa.array(arr_3d.reshape(-1), type=dtype)
            inner = pa.FixedSizeListArray.from_arrays(flat, W)
            outer = pa.FixedSizeListArray.from_arrays(inner, H)

            col_idx = table.column_names.index("observation.top_depth")
            table = table.remove_column(col_idx)
            table = table.add_column(col_idx, "observation.top_depth", outer)
            pq.write_table(table, pf)
            print(f"  normalized {pf.name}")

        except Exception as e:
            print(f"  failed {pf}: {e}")
            continue

    print(f"[done] {Path(dataset_root).name}")

dp._fix_depth_data_format = _safe_fix_depth

# --- rest of your original script unchanged ---
from scripts.utils.dataset_processing import merge_datasets

release_dirs = [
    Path("Datasets/example/record_pant_long_release_10"),
    Path("Datasets/example/record_pant_short_release_10"),
    Path("Datasets/example/record_top_long_release_10"),
    Path("Datasets/example/record_top_short_release_10"),
]

source_roots = []
for release_dir in release_dirs:
    if release_dir.exists():
        for sub in release_dir.iterdir():
            if sub.is_dir() and (sub / "meta").exists():
                source_roots.append(sub)

output_root = Path("Datasets/example/four_types_merged_with_depth")

merge_datasets(
    source_roots=source_roots,
    output_root=output_root,
    output_repo_id="four_types_merged_with_depth",
    merge_custom_meta=True,
)