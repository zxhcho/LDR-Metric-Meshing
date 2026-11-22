# Learning Discrete Riemannian Metric for Problem Aware Meshing

[English](README.md)

论文《Learning Discrete Riemannian Metric for Problem Aware Meshing》
（Zhang, Jin, Bao, Huang，*Computer Graphics Forum*，2026）中表格
`tab:obstacle_convergence_tab` 的公开代码。
DOI：[10.1111/cgf.70606](https://doi.org/10.1111/cgf.70606)。

## 引用

Wiley Online Library 的导出。`abstract` 字段已去掉。

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

## 数据

权重和测试载荷是归档 `obstacle-40k_obs-predict.tar`，不在 git 中。下载：

<https://drive.google.com/file/d/1Ze4hCkDV_HizTEzq86KrjfRZ_1xwe0E2/view?usp=drive_link>

该归档为 1,612,779,520 字节，MD5 为 `cca74919983b639589c38629c19cf57e`。
`data/MANIFEST.yaml` 记录同一文件名、链接、字节数和 MD5。解压到一个空目录：

```
mkdir -p "$ARCHIVE_ROOT"
tar -C "$ARCHIVE_ROOT" -xf obstacle-40k_obs-predict.tar
```

```
checkpoint/model_100.pth          第 100 轮权重
infer/samples/*.npz               100 个网络输入
infer/op_cache/plane_44k.npz      DiffusionNet 算子
mesh/plane_44k.obj                背景网格
field/obstacle/*.vtk              障碍场
field/init/*.vtk                  求解初值
field/gt/*.vtk                    相对 L1 的真值
reference/edge/*.vtk              参考预测边长
log/ours_amr.yaml                 Ours，带 AMR
log/ours.yaml                     Ours，不带 AMR
log/obs_amr.yaml                  障碍度量，带 AMR
log/obs.yaml                      障碍度量，不带 AMR
log/uni4k_amr.yaml                均匀 4k，带 AMR
log/uni4k.yaml                    均匀 4k，不带 AMR
log/uni12k.yaml                   均匀 12k，不带 AMR
```

`Obs` 与 `Uni` 读取归档之外的 `$BASELINE_ROOT`：

```
obs_metric/*.mat
uni4k_mesh/*.vtk
uni4k_init/*.vtk
uni12k_mesh/*.vtk
uni12k_init/*.vtk
```

## 推理

`docker build -t metriccovector-obstacle:infer -f docker/Dockerfile .` 基于
`ubuntu:22.04`，安装 C/C++ 工具链、CMake、git、SCOTCH、SuiteSparse CHOLMOD 和
PyTorch `1.13.1+cu117`。下面的命令编译求解器，把归档链接成 `pipeline.run`
读取的路径，把预测边长写到 `/tmp/work`，并与 `reference/edge` 比较。

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

`METRIC_BIN_ROOT` 是 `field_projection`、`field_metric_ref`、`discrete_error`、
`metric_from_edge`、`mmg_2d_metric`、`ex36_mesh`、`ex36_mesh_AMR` 所在目录。
容器内该目录是 `/tmp/metric-bin`。

## 表格

`scripts/run_obstacle_table.py` 依次执行推理、`metric_from_edge`、
`mmg_2d_metric`（`dof=4000`、`hmin=0.1`、`hmax=10`）、`field_projection`、
`ex36_mesh`（`max-it=100`、`tol=0.001`、`step-tol=1e-5`）和
`ex36_mesh_AMR`（`max-amr=100`、`max-it=1`，容差相同）。每个求解进程使用一个线程。
结果写到 `--work-root`。

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

不传 `--baseline-root` 时只跑两组 Ours。边长 VTK 已在 `$WORK_ROOT` 时加上
`--skip-infer`。

## 检查

```
python3 scripts/check_table3.py --data-root "$ARCHIVE_ROOT"
python3 scripts/check_dataset_shape.py --data-root "$ARCHIVE_ROOT"
```

`check_table3.py` 从 7 个日志重算表 3 的印刷值。`check_dataset_shape.py` 检查
`mesh/plane_44k.obj` 的 44,304 个顶点和 87,672 个面、每份日志的 100 个样本，以及
`reference/edge` 的 100 个 `.vtk`。`--data-root` 指向解压后的归档。
