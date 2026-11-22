import argparse
import yaml
import os
import numpy as np


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--aniso", type=str)
    args.add_argument("--iso", type=str)
    args.add_argument("--threshold", type=float, default=1.0e-1)
    args = args.parse_args()

    aniso_log_file = args.aniso
    iso_log_file = args.iso
    iso_suffix = os.path.basename(iso_log_file).replace(".yaml", "")
    record_file = aniso_log_file.replace(".yaml", f"_iso_{iso_suffix}")

    with open(aniso_log_file, 'r') as f:
        data_aniso = yaml.safe_load(f)

    with open(iso_log_file, 'r') as f:
        data_iso = yaml.safe_load(f)

    n = len(data_aniso)
    assert n == len(data_iso)

    error_measure_list = ['c0_rel_error_l1',
                          'c0_rel_error_l2', 'c1_rel_error_l1', 'c1_rel_error_l2']

    for error_measure in error_measure_list:
        record_error_file = f"{record_file}_{error_measure}.log"
        data_diff = {}
        n_success = 0
        n_fail = 0
        mean_dof = 0
        aniso_errors = []
        iso_errors = []

        not_converged = []
        threshold = args.threshold
        with open(record_error_file, 'w') as f:
            for i in range(n):
                key = list(data_aniso.keys())[i]
                error_aniso = data_aniso[key][error_measure]
                error_iso = data_iso[key][error_measure]
                mean_dof += data_aniso[key]['DOF']
                if error_aniso > threshold or i in not_converged:
                    f.write(f"{key}: failed for not converged\n")
                    n_fail += 1
                    continue
                data_diff[key] = error_aniso - error_iso
                aniso_errors.append(error_aniso)
                iso_errors.append(error_iso)
                if error_aniso < error_iso:
                    n_success += 1
                else:
                    n_fail += 1
                    f.write(
                        f"{key}: failed. iso: {error_iso}, aniso: {error_aniso}\n")
            aniso_errors = np.array(aniso_errors)
            iso_errors = np.array(iso_errors)
            f.write(f"success: {n_success}/{n}, fail: {n_fail} \n")
            f.write(
                f"aniso mean: {np.mean(aniso_errors)}, dof: {mean_dof/n} \n")
            f.write(f"iso mean: {np.mean(iso_errors)} \n")
            f.write(
                f"aniso iso mean ratio: {np.mean(aniso_errors) / np.mean(iso_errors)} \n")
            f.write(
                f"aniso/iso ratio mean: {np.mean(aniso_errors/iso_errors)} \n")
