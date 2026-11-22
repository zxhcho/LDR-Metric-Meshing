import torch

from criterion.build_criterion import CRITERION_REGISTRY
from utils.stat import merge_mean, merge_std


def get_ff_uv(b1, b2, ff):
    ff_u = torch.einsum('ij,ij->i', ff, b1)
    ff_v = torch.einsum('ij,ij->i', ff, b2)
    ff_uv = torch.stack((ff_u, ff_v), dim=1)
    return ff_uv


def cal_metric_tensor_from_ortho_frames(frame: torch.Tensor):
    """
    args:
        frame: (n, k, k), n is batch size, k is dimension
    rtn:
        metric: (n, k, k)
    """
    metric = torch.matmul(frame.transpose(1, 2), frame)
    metric_inv = torch.inverse(metric)
    return metric_inv


def inner_product_with_metric(metric: torch.Tensor,
                              vec1: torch.Tensor,
                              vec2: torch.Tensor):
    """
    return vec1^T * metric * vec2
    args:
        metric: (n, k, k)
        vec1: (n, k)
        vec2: (n, k)
    rtn:
        product: (n)
    """
    n, k = vec1.shape

    vec1 = vec1.reshape(n, k, 1)
    vec2 = vec2.reshape(n, k, 1)

    product = torch.matmul(
        torch.matmul(vec1.transpose(1, 2), metric), vec2).squeeze()
    return product


def norm_with_metric(metric: torch.Tensor,
                     vec: torch.Tensor):
    """
    get vector norm respect to a metric
    args:
        metric: (n, k, k)
        vec: (n, k)
    rtn:
        norm: (n)
    """
    product = inner_product_with_metric(metric, vec, vec)
    product = torch.clip(product, 1e-8, None)
    norm = torch.sqrt(product)

    return norm


def angle_with_metric(metric: torch.Tensor,
                      vec1: torch.Tensor,
                      vec2: torch.Tensor):
    """
    calculate angle of 2 vectors respect to an angle
    args:
        metric: (n, k, k)
        vec1: (n, k)
        vec2: (n, k)
    rtn:
        angle: (n)
    """
    product = inner_product_with_metric(metric, vec1, vec2)
    norm1 = norm_with_metric(metric, vec1)
    norm2 = norm_with_metric(metric, vec2)
    cos = product / norm1 / norm2
    cos = torch.clip(cos, -1, 1)
    angle = torch.arccos(cos)
    return angle


def calc_guassian_curv_from_metrics(verts: torch.Tensor,
                                    faces: torch.Tensor,
                                    b1: torch.Tensor,
                                    b2: torch.Tensor,
                                    metrics: torch.Tensor):
    """
    get guassian curvature of each point respect to piecewise constant metrics on triangle
    args:
        verts: (nv, 3)
        faces: (nf, 3)
        b1: (nf, 3)
        b2, (nf, 3)
        metrics: (nf, 2, 2)
    rtn:
        guassian_curv: (nv)
    """
    nv = verts.shape[0]
    e1 = verts[faces[:, 1]] - verts[faces[:, 0]]
    e1 = get_ff_uv(b1, b2, e1)
    e2 = verts[faces[:, 2]] - verts[faces[:, 1]]
    e2 = get_ff_uv(b1, b2, e2)
    e3 = e1+e2
    norm1 = norm_with_metric(metrics, e1)
    norm2 = norm_with_metric(metrics, e2)
    norm3 = norm_with_metric(metrics, e3)
    inner1 = inner_product_with_metric(metrics, e1, e3)
    inner2 = inner_product_with_metric(metrics, -e1, e2)
    inner3 = inner_product_with_metric(metrics, e2, e3)
    eps = 1e-7
    left = -1+eps
    right = 1-eps
    angle1 = torch.arccos(torch.clip(inner1 / norm1 / norm3, left, right))
    angle2 = torch.arccos(torch.clip(inner2 / norm1 / norm2, left, right))
    angle3 = torch.arccos(torch.clip(inner3 / norm2 / norm3, left, right))
    curv = torch.zeros((nv)).to(verts.device)
    curv = curv.scatter_reduce(0, faces[:, 0], angle1, reduce="sum")
    curv = curv.scatter_reduce(0, faces[:, 1], angle2, reduce="sum")
    curv = curv.scatter_reduce(0, faces[:, 2], angle3, reduce="sum")
    curv = torch.nan_to_num(curv, nan=0.0)
    return 2*torch.pi - curv


@CRITERION_REGISTRY("MetricGaussian")
def framefield_metric_guassian(pred: torch.Tensor, **data):
    """
        pred: [N, 4]
    """
    verts = data["verts"]
    faces = data["faces"]
    local_frame = data["local_frame"]
    boundary_idx = data["boundary_idx"]
    gt_curv = data["curv"]
    b1 = local_frame[:, :3]
    b2 = local_frame[:, 3:6]

    pred = pred.reshape(-1, 2, 2)
    pred_metrics = torch.bmm(pred.transpose(1, 2), pred)

    pred_curv = calc_guassian_curv_from_metrics(
        verts, faces, b1, b2, pred_metrics)
    pred_curv[boundary_idx] = 0
    pred_curv /= torch.pi/2

    return torch.nn.functional.mse_loss(gt_curv, pred_curv)


@CRITERION_REGISTRY("MetricMSE")
def framefield_metric_guassian(pred: torch.Tensor, **data):
    """
        pred: [N, 4]
    """
    pred_j = pred.reshape(-1, 2, 2)
    pred_metrics = torch.bmm(pred_j.transpose(1, 2), pred_j)
    pred_metrics = pred_metrics.reshape(-1, 4)
    gt_metric = data['metrics']
    return torch.nn.functional.mse_loss(pred_metrics, gt_metric)


def logits_to_eig_uv(pred: torch.Tensor):
    """
        pred: [N, 4]
        eig_uv: [N, 4]
    """

    vec = pred[:, :2]
    vec = vec / (torch.norm(vec, dim=-1, keepdim=True) + 1e-8)
    vec_orth = torch.stack((-vec[:, 1], vec[:, 0]), dim=1)

    eig_uv = torch.cat([vec * pred[:, 2].unsqueeze(1),
                        vec_orth * pred[:, 3].unsqueeze(1)], dim=1)
    return eig_uv


def logits_to_metric(pred: torch.Tensor):
    """
        pred: [N, 4]
        metric: [N, 2, 2]
    """
    eig_uv = logits_to_eig_uv(pred).reshape(-1, 2, 2)
    metrics_uv = eig_uv.transpose(1, 2) @ eig_uv
    return metrics_uv


@CRITERION_REGISTRY("MetricLen")
def metric_len(pred: torch.Tensor, **data):
    """
        pred: [N, 4]
    """
    ef = data["ef"]  # -1 must be removed from it
    edge_vec_uv = data["edge_vec_uv"]  # (2*ne, 2), set as 0 on boundary
    edge_weight = data["edge_weight"]  # (ne)

    ne = ef.shape[0]
    edge_vec_uv_pos = edge_vec_uv[:ne, :]
    edge_vec_uv_neg = edge_vec_uv[ne:, :]

    pred_metrics = logits_to_metric(pred)  # (nf, 2, 2)
    metrics_edge_pos = torch.index_select(
        pred_metrics, 0, ef[:, 0])  # (ne, 2, 2)
    metrics_edge_neg = torch.index_select(
        pred_metrics, 0, ef[:, 1])  # (ne, 2, 2)

    norm_pos = norm_with_metric(metrics_edge_pos, edge_vec_uv_pos)
    norm_neg = norm_with_metric(metrics_edge_neg, edge_vec_uv_neg)

    edge_diff = torch.abs(norm_pos - norm_neg) * edge_weight

    return torch.mean(edge_diff)


def swap_metric(pred: torch.Tensor, mask=None):
    """
        pred: [N, 4]
    """
    metric_swap = pred.clone()

    if mask == None:
        metric_swap[:, 0] = -pred[:, 1]
        metric_swap[:, 1] = pred[:, 0]
        metric_swap[:, 2] = pred[:, 3]
        metric_swap[:, 3] = pred[:, 2]
    else:
        metric_swap[mask, 0] = -pred[mask, 1]
        metric_swap[mask, 1] = pred[mask, 0]
        metric_swap[mask, 2] = pred[mask, 3]
        metric_swap[mask, 3] = pred[mask, 2]

    return metric_swap


def angle_loss(pred: torch.Tensor,
               gt: torch.Tensor):
    cos_val = torch.sum(pred * gt, dim=-1)
    exp_val = 1 - torch.exp(2*torch.square(cos_val) - 2)
    return exp_val


def rel_size_loss(pred: torch.Tensor,
                  gt: torch.Tensor):
    loss = (torch.abs(pred - gt) / gt).mean(dim=-1)
    return loss


def composed_loss(pred: torch.Tensor,
                  gt: torch.Tensor,
                  dir_weight=1.0,
                  size_weight=1.0):
    pred_dir = pred[:, :2]
    gt_dir = gt[:, :2]
    pred_dir = pred_dir / (torch.norm(pred_dir, dim=-1, keepdim=True) + 1e-8)

    pred_size = pred[:, 2:]
    gt_size = gt[:, 2:]
    return angle_loss(pred_dir, gt_dir) * dir_weight + rel_size_loss(pred_size, gt_size) * size_weight


@CRITERION_REGISTRY("MetricEig")
def metric_eig(pred: torch.Tensor, **data):
    """
        pred: [N, 4]
    """
    gt = data["metrics_eig_ortho"]

    n = gt.size(0)
    losses = torch.empty([2, n], dtype=pred.dtype, device=pred.device)

    losses[0] = composed_loss(pred, gt)
    losses[1] = composed_loss(swap_metric(pred), gt)

    min_idx = torch.argmin(losses, dim=0)
    mask = min_idx > 0

    pred_sym = swap_metric(pred, mask)
    loss = composed_loss(pred_sym, gt)

    return loss.mean()


def metric_measure(pred: torch.Tensor,
                   gt: torch.Tensor):
    pred_dir = pred[:, :2]
    gt_dir = gt[:, :2]

    cos_sim = torch.cosine_similarity(pred_dir, gt_dir, dim=1)
    angle_diff = torch.rad2deg_(torch.acos_(
        torch.abs_(torch.clamp_(cos_sim, -1, 1))))

    pred_size = pred[:, 2:]
    gt_size = gt[:, 2:]
    rel_size_diff = (torch.abs(pred_size - gt_size) / gt_size).mean(dim=-1)

    return angle_diff, rel_size_diff


def metric_measure_sym(pred: torch.Tensor,
                       gt: torch.Tensor):
    pred_swap = swap_metric(pred)
    angle_diff, rel_size_diff = metric_measure(pred, gt)
    angle_diff_swap, rel_size_diff_swap = metric_measure(pred_swap, gt)

    angle_diff = torch.minimum(angle_diff, angle_diff_swap)
    rel_size_diff = torch.minimum(rel_size_diff, rel_size_diff_swap)

    angle_std, angle_mean = torch.std_mean(angle_diff)
    rel_size_std, rel_size_mean = torch.std_mean(rel_size_diff)

    measure = {
        "num": 1.0,
        "angle_mean": angle_mean,
        "angle_std": angle_std,
        "rel_size_mean": rel_size_mean,
        "rel_size_std": rel_size_std,
    }
    return measure


def merge_measure(m1, m2):
    if m1["num"] == 0:
        return m2
    elif m2["num"] == 0:
        return m1

    metric = {}
    measures = ["angle", "rel_size"]
    metric["num"] = m1["num"] + m2["num"]
    for m in measures:
        key_mean = f"{m}_mean"
        key_std = f"{m}_std"
        metric[key_mean] = merge_mean(
            m1["num"], m1[key_mean], m2["num"], m2[key_mean])
        metric[key_std] = merge_std(
            m1["num"], m1[key_mean], m1[key_std], m2["num"], m2[key_mean], m2[key_std])

    return metric


@CRITERION_REGISTRY("DiscreteMetric")
def metric_eig(pred: torch.Tensor, **data):
    """
        pred: [ne, 1]
    """
    gt = data["length"].reshape(-1)
    pred = pred.reshape(-1)

    rel_error = torch.abs((pred - gt) / (gt + 1e-8))
    return rel_error.mean()


@CRITERION_REGISTRY("DiscreteMetricAbs")
def metric_eig(pred: torch.Tensor, **data):
    """
        pred: [ne, 1]
    """
    gt = data["length"].reshape(-1)
    pred = pred.reshape(-1)

    return torch.nn.functional.mse_loss(pred, gt)


@CRITERION_REGISTRY("MatrixMetricAbs")
def metric_eig(pred: torch.Tensor, **data):
    """
        pred: [ne, 1]
    """
    gt = data["metric"].reshape(-1)
    pred = pred.reshape(-1)

    return torch.nn.functional.mse_loss(pred, gt)
