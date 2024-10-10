import h5py
import numpy as np
import os


def average_data(results_path, times):
    test_acc = get_all_results_for_one_algo(results_path, times)

    max_accurancy = []
    for i in range(times):
        max_accurancy.append(test_acc[i].max())

    acc_mean = np.mean(max_accurancy)
    acc_std = np.std(max_accurancy)
    print("std for best accurancy:", acc_std)
    print("mean for best accurancy:", acc_mean)

    return acc_mean, acc_std


def get_all_results_for_one_algo(results_path, times):
    test_acc = []
    for i in range(times):
        file_name = os.path.join(results_path, 'results_{}.h5'.format(i))
        test_acc.append(np.array(read_data_then_delete(file_name, delete=False)))

    return test_acc


def read_data_then_delete(file_name, delete=False):
    with h5py.File(file_name, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))

    if delete:
        os.remove(file_name)
    print("Length: ", len(rs_test_acc))

    return rs_test_acc