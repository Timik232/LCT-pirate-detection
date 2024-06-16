from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt
import numpy as np

def make_plt_rows(matrix_l, plt_verbose=False):
    """
    Make plot of rows
    :param plt_verbose:
    :param matrix_l:
    :return:
    """
    points_dict = {}

    for i in range(max(matrix_l.shape[0], matrix_l.shape[1])):
        points_dict[i] = 0

    for i in range(matrix_l.shape[0]):
        max_index = np.argmax(matrix_l[i])
        max_value = matrix_l[i, max_index]
        points_dict[max_index] = max_value

    x_points = list(points_dict.keys())
    y_points = list(points_dict.values())
    y_points = np.maximum(y_points, 0)

    sorted_indices = np.argsort(x_points)
    x_points = np.take(np.array(x_points), sorted_indices)
    y_points = np.take(np.array(y_points), sorted_indices)

    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='same')

    window_size = max(int(y_points.shape[0] * 0.05), 3)
    y_points_smoothed = moving_average(y_points, window_size)

    f_interp = PchipInterpolator(x_points, y_points_smoothed)
    x_smooth = np.linspace(x_points.min(), x_points.max(), len(x_points))
    y_smooth = f_interp(x_smooth)

    peaks, _ = find_peaks(y_smooth)
    if len(peaks) == 0:
        return {"interval" : ""}
    widths_half_max = peak_widths(y_smooth, peaks, rel_height=0.50)

    max_peak_idx = np.argmax(y_smooth[peaks])
    max_peak_height = y_smooth[peaks][max_peak_idx]

    max_width_idx = np.argmax(widths_half_max[0])
    max_peak_width = widths_half_max[0][max_width_idx]
    left_ips_x = 0
    right_ips_x = 0
    if peaks[max_peak_idx] == peaks[max_width_idx]:
        left_ips_x = x_smooth[int(widths_half_max[2][max_width_idx])]
        right_ips_x = x_smooth[int(widths_half_max[3][max_width_idx])]
    elif peaks[max_width_idx] and widths_half_max[0][max_width_idx] > 10 and y_smooth[peaks][max_width_idx] > 0.3:
        left_ips_x = x_smooth[int(widths_half_max[2][max_peak_idx])]
        right_ips_x = x_smooth[int(widths_half_max[3][max_peak_idx])]
    else:
        left_ips_x = 0
        right_ips_x = 0
        return {"interval" : ""}
    if plt_verbose:
        plt.plot(x_smooth, y_smooth)
        plt.xlabel('Index of Minimum Cosine Distance')
        plt.ylabel('Sum of Max Value - Min Value')
        plt.title('Graph of Minimum Cosine Distances with Peaks')
        plt.grid(True)
        plt.show()
    return {"interval": f"{left_ips_x}-{right_ips_x}", "width": widths_half_max[0][max_width_idx],
            "height": y_smooth[peaks][max_width_idx]}


def make_plt_columns(matrix_l, plt_verbose=False):
    """
    Make plot of columns
    :param plt_verbose:
    :param matrix_l:
    :return:
    """
    points_dict = {}
    for j in range(max(matrix_l.shape[0], matrix_l.shape[1])):
        points_dict[j] = 0
    print(matrix_l.shape[1])
    for j in range(matrix_l.shape[1]):
        max_index = np.argmax(matrix_l[:, j])
        max_value = matrix_l[max_index, j]
        points_dict[max_index] = max_value

    x_points = list(points_dict.keys())
    y_points = list(points_dict.values())

    sorted_indices = np.argsort(x_points)
    x_points = np.take(np.array(x_points), sorted_indices)
    y_points = np.take(np.array(y_points), sorted_indices)

    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='same')

    window_size = max(int(y_points.shape[0] * 0.05), 3)
    y_points_smoothed = moving_average(y_points, window_size)

    f_interp = PchipInterpolator(x_points, y_points_smoothed)
    x_smooth = np.linspace(x_points.min(), x_points.max(), len(x_points))
    y_smooth = f_interp(x_smooth)

    peaks, _ = find_peaks(y_smooth)
    if len(peaks) == 0:
        return {"interval" : ""}
    widths_half_max = peak_widths(y_smooth, peaks, rel_height=0.50)

    max_peak_idx = np.argmax(y_smooth[peaks])
    max_peak_height = y_smooth[peaks][max_peak_idx]

    max_width_idx = np.argmax(widths_half_max[0])
    max_peak_width = widths_half_max[0][max_width_idx]
    left_ips_x = 0
    right_ips_x = 0
    if peaks[max_peak_idx] == peaks[max_width_idx]:
        left_ips_x = x_smooth[int(widths_half_max[2][max_width_idx])]
        right_ips_x = x_smooth[int(widths_half_max[3][max_width_idx])]
    elif peaks[max_width_idx] and widths_half_max[0][max_width_idx] > 10 and y_smooth[peaks][max_width_idx] > 0.3:
        left_ips_x = x_smooth[int(widths_half_max[2][max_peak_idx])]
        right_ips_x = x_smooth[int(widths_half_max[3][max_peak_idx])]
    else:
        left_ips_x = 0
        right_ips_x = 0
        return {"interval" : ""}
    if plt_verbose:
        plt.plot(x_smooth, y_smooth)
        plt.xlabel('Index of Minimum Cosine Distance')
        plt.ylabel('Sum of Max Value - Min Value')
        plt.title('Graph of Minimum Cosine Distances with Peaks')
        plt.grid(True)
        plt.show()
    return {"interval": f"{left_ips_x}-{right_ips_x}", "width": widths_half_max[0][max_width_idx],
            "height": y_smooth[peaks][max_width_idx]}