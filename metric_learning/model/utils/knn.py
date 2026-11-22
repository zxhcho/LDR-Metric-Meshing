import torch
# from pytorch3d.ops import knn_points


# def knn_query(
#         pos: torch.FloatTensor,
#         query: torch.FloatTensor,
#         k: int,
#         offset=0):
#     """
#     :param  pos:     (B, N, C)
#     :param  query:   (B, M, C)
#     :return knn_idx: (B, M, k)
#     """
#     _, knn_idx, _ = knn_points(pos, query, K=k+offset, return_nn=False)
#     return knn_idx[:, :, offset:]


def knn_group(x: torch.FloatTensor,
              idx: torch.LongTensor,
              relative_feat=False,
              relative_feat_append=False):
    """
    extract local patch
    args:
        x:      (N, C)
        idx:    (M, k)
    rtn:
        if relative_feat_append:
            (M, k, 2*C)
        else:
            (M, k, C)
    """
    c = x.size(1)
    m, k = idx.size(0), idx.size(1)
    x = torch.index_select(x, 0, idx.reshape(-1))
    x = x.reshape(m, k, c)
    if relative_feat:
        center = x[:, 0, :].unsqueeze(1)
        x = x - center
        if relative_feat_append:
            x = torch.cat([x, center.repeat(1, k, 1)], dim=-1)

    return x


def pairwise_distance(x):
    """
    compute pairwise distance of a point cloud
    args:
        x: (N, C)
    rtn:
        (N, N)
    """
    # pairwise inner product
    inner = torch.matmul(x, x.transpose(0, 1))  # (N, C) -> (N, N)

    # norm of each point in feature space
    xx = torch.sum(x**2, dim=1, keepdim=True)  # (N, C) -> (N, 1)

    # (N, N), negative pairwise distance
    pairwise_distance = -(xx + xx.transpose(0, 1) - 2 * inner)
    return pairwise_distance


def knn_feature(x, k=20):
    """
    knn in feature space
    args:
        x: (N, C)
        K: int
    rtn:
        idx: (N, K)
    """
    # pairwise inner product
    inner = torch.matmul(x, x.transpose(0, 1))  # (N, C) -> (N, N)

    # norm of each point in feature space
    xx = torch.sum(x**2, dim=1, keepdim=True)  # (N, C) -> (N, 1)

    # (N, N), negative pairwise distance
    pairwise_distance = -(xx + xx.transpose(0, 1) - 2 * inner)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


if __name__ == '__main__':
    torch.manual_seed(0)

    pos = torch.rand(10, 3)
    idx = torch.randint(0, 10, (5, 4))

    # x = knn_group(pos, idx)
    # x_relative = knn_group(pos, idx, relative_feat=True)
    # x_relative_1 = knn_group(pos, idx, relative_feat=True, relative_dim=1)

    # print("pos:", pos)
    # print("idx:", idx)
    # print("x:", x)
    # print("x_relative:", x_relative)
    # print("x_relative_1:", x_relative_1)

    import time
    start_time = time.time()
    dist = pairwise_distance(pos)
    print("--- %s ms ---" % (1000*(time.time() - start_time)))

    from sklearn.metrics.pairwise import euclidean_distances
    start_time = time.time()
    dist = -euclidean_distances(pos, squared=True)
    print("--- %s ms ---" % (1000*(time.time() - start_time)))
