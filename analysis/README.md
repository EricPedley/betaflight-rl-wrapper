# Blackbox Analysis Tools

## Setup

`blackbox-tools` should be cloned as a submodule here.
To set it up, run `make` from inside the repo and it'll generate the binary `obj/blackbox_decode` which takes a .bbl file and spits out a csv.

```bash
cd analysis/blackbox-tools
make
```

## Usage

### analyze_blackbox.py

Comprehensive blackbox analysis script that decodes .bbl files and visualizes them in Rerun with:
- All blackbox fields logged as time series
- Raw RC channel values
- Semantic RC channel names and transformed values (as passed to policy observations)
- Drone pose visualization using `log_drone_pose` from the sitl package
- Full 12-dimensional policy observation vector computed from state

```bash
python analysis/analyze_blackbox.py <path_to_bbl_file>
```

The script will:
1. Copy the .bbl file to /tmp
2. Run blackbox_decode to generate CSV
3. Parse and log all data to Rerun with semantic organization

**Requirements:**
```bash
pip install rerun-sdk numpy scipy
```

### blackbox_to_rerun.py

Legacy script for basic CSV log visualization. Use `analyze_blackbox.py` for .bbl files directly.

```bash
python analysis/blackbox_to_rerun.py <path_to_csv_file>
```