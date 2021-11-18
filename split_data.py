import numpy as np


def split_hist_future(arr, hist_win, future_win, slide_win):
    # arr: m*n
    # m: number of timepoints
    # n: number of features
    arr = np.array(arr)
    if len(arr.shape) == 1:
        arr = arr.reshape(arr.shape[0], 1)
    n_timepoints = arr.shape[0]
    t_start = hist_win
    t_end = n_timepoints - future_win
    hist = []
    future = []
    for t in range(t_start, t_end, slide_win):
        hist.append(arr[t-hist_win:t, :])
        future.append(arr[t:t+future_win, :])
    return np.array(hist), np.array(future)


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
