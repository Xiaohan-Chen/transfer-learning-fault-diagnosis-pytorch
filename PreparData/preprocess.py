import numpy as np

def transformation(sub_data, fft, normalization, backbone):

    if fft:
        sub_data = np.fft.fft(sub_data)
        sub_data = np.abs(sub_data) / len(sub_data)
        sub_data = sub_data[:int(sub_data.shape[0] / 2)].reshape(-1,)                

    if normalization == "0-1":
        sub_data = (sub_data - sub_data.min()) / (sub_data.max() - sub_data.min())
    elif normalization == "mean-std":
        sub_data = (sub_data - sub_data.mean()) / sub_data.std()

    if backbone in ("ResNet1D", "CNN1D"):
        sub_data = sub_data[np.newaxis, :]
    elif backbone == "ResNet2D":
        n = int(np.sqrt(sub_data.shape[0]))
        if fft:
            sub_data = sub_data[:n*n]
        sub_data = np.reshape(sub_data, (n, n))
        sub_data = sub_data[np.newaxis, :]
        sub_data = np.concatenate((sub_data, sub_data, sub_data), axis=0)

    return sub_data