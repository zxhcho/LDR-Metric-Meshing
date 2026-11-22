import torch
import subprocess


def merge_mean(x_size, x_mean, y_size, y_mean):
    return (x_size * x_mean + y_size * y_mean) / (x_size + y_size)


def merge_std(x_size, x_mean, x_var, y_size, y_mean, y_var):
    return (x_size * (x_var + (x_mean ** 2)) + y_size * (y_var + (y_mean ** 2))) / (x_size + y_size) - (((x_size * x_mean + y_size * y_mean) / (x_size + y_size)) ** 2)


def dic_2_csv(dic, csv_path):
    with open(csv_path, 'w') as f:
        for key in dic.keys():
            f.write("{},{:.6f}\n".format(key, dic[key]))


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


if __name__ == "__main__":
    n = 10000000000
    mat = torch.rand(n)
    print("generate mat: ", mat.shape)
    mean, var = torch.std_mean(mat)
    print("torch mean: ", mean)
    print("torch var: ", var)

    batch_size = 10000
    num_batch = n // batch_size
    n_prev = 0
    mean_accumulate = 0
    var_accumulate = 0
    for i in range(num_batch):
        batch = mat[i * batch_size:(i + 1) * batch_size]
        mean_batch, var_batch = torch.std_mean(batch)
        mean_accumulate = merge_mean(
            n_prev, mean_accumulate, batch_size, mean_batch)
        var_accumulate = merge_std(
            n_prev, mean_accumulate, var_accumulate, batch_size, mean_batch, var_batch)
        n_prev += batch_size

    print("accumulate mean: ", mean_accumulate)
    print("accumulate var: ", var_accumulate)
