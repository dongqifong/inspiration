import numpy as np


def split_his_furture(arr, his_win, fur_win, slide_win):
    # arr: m*n
    # m: number of timepoints
    # n: number of features
    arr = np.array(arr)
    if len(arr.shape) == 1:
        arr = arr.reshape(arr.shape[0], 1)
    n_timepoints = arr.shape[0]
    t_start = his_win
    t_end = n_timepoints - fur_win
    his = []
    future = []
    for t in range(t_start, t_end, slide_win):
        his.append(arr[t-his_win:t, :])
        future.append(arr[t:t+fur_win, :])
    return np.array(his), np.array(future)


def train_test_split(x, y=None, train_size=0.66, random_state=0, shuffle=True):
    from sklearn.model_selection import train_test_split
    if y is not None:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=train_size, random_state=random_state, shuffle=shuffle)
        return x_train, x_test, y_train, y_test
    else:
        x_train, x_tset = train_test_split(
            x, train_size=train_size, random_state=random_state, shuffle=shuffle)
        return x_train, x_tset
