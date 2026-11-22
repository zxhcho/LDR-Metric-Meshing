# Learning Discrete Riemannian Metric for Problem Aware Meshing

[中文](README.zh.md)

Public code for table `tab:obstacle_convergence_tab` in Zhang, Jin, Bao, and
Huang, *Computer Graphics Forum*, 2026,
DOI [10.1111/cgf.70606](https://doi.org/10.1111/cgf.70606).

## Citation

Wiley Online Library export. The `abstract` field is omitted.

```bibtex
@article{https://doi.org/10.1111/cgf.70606,
  author = {Zhang, Xiaohu and Jin, Yiyang and Bao, Hujun and Huang, Jin},
  title = {Learning Discrete Riemannian Metric for Problem Aware Meshing},
  journal = {Computer Graphics Forum},
  volume = {n/a},
  number = {n/a},
  pages = {e70606},
  keywords = {CCS Concepts, • Computing methodologies → Mesh geometry models, Riemannian metric, Mesh adaptation, Anisotropic meshing},
  doi = {https://doi.org/10.1111/cgf.70606},
  url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.70606},
  eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1111/cgf.70606}
}
```

## Data

The weight and the test payload are the archive
`obstacle-40k_obs-predict.tar`. They are not in git. Download:

<https://drive.google.com/file/d/1Ze4hCkDV_HizTEzq86KrjfRZ_1xwe0E2/view?usp=drive_link>

The archive is 1,612,779,520 bytes. Its MD5 is
`cca74919983b639589c38629c19cf57e`. `data/MANIFEST.yaml` records that file
name, URL, byte count, and MD5. Unpack into an empty directory:

```
mkdir -p "$ARCHIVE_ROOT"
tar -C "$ARCHIVE_ROOT" -xf obstacle-40k_obs-predict.tar
```

```
checkpoint/model_100.pth          epoch-100 weight
infer/samples/*.npz               100 network inputs
infer/op_cache/plane_44k.npz      DiffusionNet operators
mesh/plane_44k.obj                background mesh
field/obstacle/*.vtk              obstacle field
field/init/*.vtk                  solver initial guess
field/gt/*.vtk                    relative-L1 ground truth
reference/edge/*.vtk              reference predicted edge lengths
log/ours_amr.yaml                 Ours, with AMR
log/ours.yaml                     Ours, without AMR
log/obs_amr.yaml                  obstacle metric, with AMR
log/obs.yaml                      obstacle metric, without AMR
log/uni4k_amr.yaml                uniform 4k, with AMR
log/uni4k.yaml                    uniform 4k, without AMR
log/uni12k.yaml                   uniform 12k, without AMR
```

The `Obs` and `Uni` arms read `$BASELINE_ROOT`, which is separate from the
archive:

```
obs_metric/*.mat
uni4k_mesh/*.vtk
uni4k_init/*.vtk
uni12k_mesh/*.vtk
uni12k_init/*.vtk
```

## Inference

`docker build -t metriccovector-obstacle:infer -f docker/Dockerfile .` builds
an `ubuntu:22.04` image with the C/C++ toolchain, CMake, git, SCOTCH,
SuiteSparse CHOLMOD, and PyTorch `1.13.1+cu117`. The run below compiles the
solvers, links the archive into the paths read by `pipeline.run`, writes
predicted edge lengths under `/tmp/work`, and compares them with
`reference/edge`.

```
docker build -t metriccovector-obstacle:infer -f docker/Dockerfile .

docker run --gpus all --rm \
  -v "$PWD":/src/release:ro \
  -v "$ARCHIVE_ROOT":/data:ro \
  -e PYTHONPATH=/src/release/metric_learning \
  metriccovector-obstacle:infer \
  bash -lc 'set -euo pipefail
    git config --global --add safe.directory /src/release
    git config --global --add safe.directory /src/release/.git
    cmake -S /src/release -B /tmp/mc-build -DCMAKE_BUILD_TYPE=Release
    cmake --build /tmp/mc-build --parallel "$(nproc)" \
      --target field_projection field_metric_ref discrete_error metric_from_edge mmg_2d_metric
    rm -rf /tmp/mfem
    git init /tmp/mfem
    git -C /tmp/mfem remote add origin https://github.com/mfem/mfem.git
    git -C /tmp/mfem fetch --depth 1 origin 8452c8cc7f6f00603f54607baf5a20842eb6915b
    git -C /tmp/mfem checkout --detach FETCH_HEAD
    git -C /tmp/mfem rev-parse HEAD | grep -q "^8452c8cc7f"
    git -C /tmp/mfem apply /src/release/libs/ex36_mesh/upstream-mfem-8452c8cc7f.patch
    rm -rf /tmp/mfem/examples/ex36_mesh
    cp -a /src/release/libs/ex36_mesh /tmp/mfem/examples/ex36_mesh
    cmake -S /tmp/mfem -B /tmp/mfem-build -DCMAKE_BUILD_TYPE=Release \
      -DFILE_IO_SOURCE_DIR=/src/release/libs/file_io \
      -DFETCHCONTENT_SOURCE_DIR_EIGEN3=/tmp/mc-build/_deps/eigen3-src \
      -DFETCHCONTENT_SOURCE_DIR_SPDLOG=/tmp/mc-build/_deps/spdlog-src \
      -DFETCHCONTENT_SOURCE_DIR_LIBIGL=/tmp/mc-build/_deps/libigl-src \
      -DFETCHCONTENT_SOURCE_DIR_YAML_CPP=/tmp/mc-build/_deps/yaml_cpp-src
    cmake --build /tmp/mfem-build --parallel "$(nproc)" --target ex36_mesh ex36_mesh_AMR
    mkdir -p /tmp/metric-bin
    cp -a /tmp/mc-build/bin/field_projection /tmp/mc-build/bin/field_metric_ref \
      /tmp/mc-build/bin/discrete_error /tmp/mc-build/bin/metric_from_edge \
      /tmp/mc-build/bin/mmg_2d_metric /tmp/metric-bin/
    find /tmp/mfem-build -type f \( -name ex36_mesh -o -name ex36_mesh_AMR \) -perm -111 \
      -exec cp -a {} /tmp/metric-bin/ \;
    python3 /src/release/scripts/release_layout.py \
      --archive-root /data --work-root /tmp/work
    export METRIC_DATA_ROOT=/tmp/work
    export METRIC_BIN_ROOT=/tmp/metric-bin
    python3 -m pipeline.run --config /src/release/metric_learning/config.example.yaml --mode infer
    python3 /src/release/scripts/compare_infer_vtk.py \
      --prediction /tmp/work/model/metric_3/predict_out/DiscreteEdge_8_128_field_Cri_DiscreteMetricAbs_1.0_Epo_100_Opt_Adam_0.0001_Sche_20_0.5/model_100.pth \
      --reference /data/reference/edge
  '
```

`METRIC_BIN_ROOT` is the directory of `field_projection`, `field_metric_ref`,
`discrete_error`, `metric_from_edge`, `mmg_2d_metric`, `ex36_mesh`, and
`ex36_mesh_AMR`. In the container that directory is `/tmp/metric-bin`.

## Table

`scripts/run_obstacle_table.py` runs inference, `metric_from_edge`,
`mmg_2d_metric` (`dof=4000`, `hmin=0.1`, `hmax=10`), `field_projection`,
`ex36_mesh` (`max-it=100`, `tol=0.001`, `step-tol=1e-5`), and
`ex36_mesh_AMR` (`max-amr=100`, `max-it=1`, same tolerances). Each solver
process uses one thread. Results are written under `--work-root`.

```
PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_obstacle_table.py \
  --repo . \
  --data-root "$ARCHIVE_ROOT" \
  --work-root "$WORK_ROOT" \
  --bin-root "$METRIC_BIN_ROOT" \
  --python python3 \
  --workers 12 \
  --baseline-root "$BASELINE_ROOT"
```

Without `--baseline-root`, the script runs the two Ours arms. Add
`--skip-infer` when the edge-length VTK files are already in `$WORK_ROOT`.

## Checks

```
python3 scripts/check_table3.py --data-root "$ARCHIVE_ROOT"
python3 scripts/check_dataset_shape.py --data-root "$ARCHIVE_ROOT"
```

`check_table3.py` recomputes the printed Table 3 cells from the seven logs.
`check_dataset_shape.py` checks 44,304 vertices and 87,672 faces in
`mesh/plane_44k.obj`, 100 samples in each log, and 100 `.vtk` files in
`reference/edge`. Pass the unpacked archive.
