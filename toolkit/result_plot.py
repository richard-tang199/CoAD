from typing import List, Dict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, MaxNLocator
from matplotlib.gridspec import GridSpec
from scipy import stats
import tqdm
from typing import Optional, Tuple
from sklearn.manifold import TSNE
import os
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from scipy.special import digamma, gamma
import warnings
warnings.filterwarnings("ignore")

matplotlib.use('agg')

__all__ = [
    "recon_plot",
    "tsne_plot",
    "get_segments",
    "estimate_entropy_knn",
    "score_plot"
]


def get_segments(label_data: np.ndarray) -> list:
    """从二值标签数组中提取异常片段的起止索引"""
    segments = []
    in_segment = False
    start = 0

    for i, label in enumerate(label_data):
        if label and not in_segment:
            start = i
            in_segment = True
        elif not label and in_segment:
            segments.append((start, i - 1))
            in_segment = False

    if in_segment:
        segments.append((start, len(label_data) - 1))

    return segments


def estimate_entropy_knn(X, k=10):
    """
    使用 k-NN 估算微分熵 (基于 Kozachenko-Leonenko 估计)
    """
    n, d = X.shape
    tree = cKDTree(X)
    # 找到第 k 个最近邻的距离
    dists, _ = tree.query(X, k=k + 1, p=2)  # k+1 因为包含自己
    r_k = dists[:, -1]  # 第 k 个邻居的距离

    # 防止 log(0)
    r_k = np.maximum(r_k, 1e-10)

    # 熵公式
    const = digamma(n) - digamma(k) + d * np.log(np.pi) - np.log(gamma(d / 2 + 1))
    entropy = const + (d / n) * np.sum(np.log(r_k))
    return entropy


def recon_plot(
        save_path: str,
        gap: float,
        test_data: np.ndarray,
        train_data: np.ndarray,
        threshold: Optional[float] = None,
        figure_length: Optional[int] = None,
        figure_width: Optional[int] = None,
        font_size: Optional[int] = None,
        recon_test_data: Optional[np.ndarray] = None,
        recon_train_data: Optional[np.ndarray] = None,
        test_anomaly_score: Optional[np.ndarray] = None,
        train_anomaly_score: Optional[np.ndarray] = None,
        test_label: Optional[np.ndarray] = None,
        train_label: Optional[np.ndarray] = None,
        plot_diff: bool = False,
        dpi: int = 100
):
    """
    绘制时间序列异常检测的可视化结果，对比训练集和测试集。

    参数:
        save_path: 保存路径（PNG文件名）
        gap: X轴刻度间隔
        test_data: 测试数据，形状为 (sequence_length, num_channels)
        train_data: 训练数据，形状为 (sequence_length, num_channels)
        threshold: 异常分数阈值线
        figure_length: 图表长度（英寸）
        figure_width: 图表宽度（英寸）
        font_size: 字体大小
        recon_test_data: 重建的测试数据
        recon_train_data: 重建的训练数据
        test_anomaly_score: 测试集异常分数
        train_anomaly_score: 训练集异常分数
        test_label: 测试集标签（0/1）
        train_label: 训练集标签（0/1）
        plot_diff: 是否绘制差异曲线
        dpi: 图像分辨率
    """
    # 数据预处理：确保数据为2D
    train_data = _ensure_2d(train_data)
    test_data = _ensure_2d(test_data)
    recon_train_data = _ensure_2d(recon_train_data) if recon_train_data is not None else None
    recon_test_data = _ensure_2d(recon_test_data) if recon_test_data is not None else None

    # 获取数据维度
    train_len, n_channels = train_data.shape
    test_len = test_data.shape[0]
    total_len = train_len + test_len

    # 自动设置图表尺寸
    if figure_length is None or figure_width is None:
        figure_length, figure_width = _auto_figure_size(total_len, n_channels)

    if font_size is None:
        font_size = max(8, min(figure_length, figure_width) // n_channels)

    # 创建子图（n_channels + 1行用于异常分数）
    n_rows = n_channels + 1
    ratio = train_len / test_len

    fig, axs = plt.subplots(
        nrows=n_rows, ncols=2,
        figsize=(figure_length, figure_width),
        tight_layout=True,
        gridspec_kw={'width_ratios': [ratio, 1]}
    )

    # 确保axs为2D数组
    if n_rows == 1:
        axs = axs.reshape(1, -1)

    # 获取异常片段
    train_segments = get_segments(train_label) if train_label is not None else None
    test_segments = get_segments(test_label) if test_label is not None else None

    # 存储差异数据用于计算异常分数
    diff_train_list = []
    diff_test_list = []

    # 绘制每个通道
    for ch in range(n_channels):
        _plot_channel(
            axs[ch], ch,
            train_data[:, ch], test_data[:, ch],
            recon_train_data[:, ch] if recon_train_data is not None else None,
            recon_test_data[:, ch] if recon_test_data is not None else None,
            train_segments, test_segments,
            gap, font_size, plot_diff,
            diff_train_list, diff_test_list
        )

    # 计算或使用异常分数
    if train_anomaly_score is None and recon_train_data is not None:
        train_anomaly_score = np.mean(diff_train_list, axis=0)

    if test_anomaly_score is None and recon_test_data is not None:
        test_anomaly_score = np.mean(diff_test_list, axis=0)

    # 绘制异常分数
    _plot_anomaly_scores(
        axs[-1],
        train_anomaly_score, test_anomaly_score,
        train_segments, test_segments,
        threshold, gap, train_len, test_len
    )

    # 保存图表
    plt.savefig(save_path, format="png", dpi=dpi, bbox_inches='tight')
    plt.close()


def _ensure_2d(data: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """确保数据为2D数组"""
    if data is None:
        return None
    return data[:, np.newaxis] if data.ndim == 1 else data


def _auto_figure_size(total_len: int, n_channels: int) -> Tuple[int, int]:
    """自动计算合适的图表尺寸"""
    if total_len >= 2000 and n_channels >= 8:
        return total_len // 100, n_channels * 5
    return 20, 15


def _plot_channel(
        ax_pair, channel_idx: int,
        train_data: np.ndarray, test_data: np.ndarray,
        recon_train: Optional[np.ndarray], recon_test: Optional[np.ndarray],
        train_segments, test_segments,
        gap: float, font_size: int, plot_diff: bool,
        diff_train_list: list, diff_test_list: list
):
    """绘制单个通道的训练和测试数据"""
    # 训练数据
    ax_train, ax_test = ax_pair

    ax_train.set_title(f"Train Data - Channel {channel_idx + 1}", fontsize=font_size, loc='left')
    ax_train.plot(train_data, label="Raw", linewidth=1)

    if recon_train is not None:
        ax_train.plot(recon_train, label="Reconstructed", linewidth=1, alpha=0.8)
        diff_train = np.abs(train_data - recon_train) * 0.8
        diff_train_list.append(diff_train)
        if plot_diff:
            ax_train.plot(diff_train, label="Difference", linewidth=1, alpha=0.6)

    _add_anomaly_highlights(ax_train, train_segments)
    _configure_axis(ax_train, len(train_data), gap, font_size)

    # 测试数据
    ax_test.set_title(f"Test Data - Channel {channel_idx + 1}", fontsize=font_size, loc='left')
    ax_test.plot(test_data, label="Raw", linewidth=1)

    if recon_test is not None:
        ax_test.plot(recon_test, label="Reconstructed", linewidth=1, alpha=0.8)
        diff_test = np.abs(test_data - recon_test) * 0.8
        diff_test_list.append(diff_test)
        if plot_diff:
            ax_test.plot(diff_test, label="Difference", linewidth=1, alpha=0.6)

    _add_anomaly_highlights(ax_test, test_segments)
    _configure_axis(ax_test, len(test_data), gap, font_size)

    # 统一Y轴范围
    y_max = max(train_data.max(), test_data.max(), 1) + 0.1
    y_min = min(train_data.min(), test_data.min(), 0) - 0.2
    ax_train.set_ylim(y_min, y_max)
    ax_test.set_ylim(y_min, y_max)


def _plot_anomaly_scores(
        ax_pair,
        train_score: Optional[np.ndarray],
        test_score: Optional[np.ndarray],
        train_segments, test_segments,
        threshold: Optional[float],
        gap: float, train_len: int, test_len: int
):
    """绘制异常分数"""
    ax_train, ax_test = ax_pair

    if train_score is not None:
        ax_train.plot(train_score, linewidth=1.5, color='blue')
        ax_train.set_title("Train Anomaly Score", fontsize=10, loc='left')
        _add_anomaly_highlights(ax_train, train_segments)
        _configure_axis(ax_train, train_len, gap, 10)

    if test_score is not None:
        ax_test.plot(test_score, linewidth=1.5, color='blue')
        ax_test.set_title("Test Anomaly Score", fontsize=10, loc='left')
        _add_anomaly_highlights(ax_test, test_segments)
        if threshold is not None:
            ax_test.axhline(y=threshold, color='red', linestyle='--',
                            linewidth=2, alpha=0.7, label=f'Threshold={threshold:.3f}')
            ax_test.legend(fontsize=8)
        _configure_axis(ax_test, test_len, gap, 10)


def _add_anomaly_highlights(ax, segments):
    """在图表上添加异常区域高亮"""
    if segments is None:
        return

    for start, end in segments:
        if start == end or end - start <= 1:
            ax.axvline(x=start, color='red', alpha=0.5, linewidth=2)
        else:
            ax.axvspan(start, end, facecolor='red', alpha=0.3)


def _configure_axis(ax, data_len: int, gap: float, font_size: int):
    """配置坐标轴"""
    ax.legend(fontsize=font_size, loc='upper right')
    ax.set_xlim(0, data_len)
    ax.grid(True, alpha=0.3, linestyle='--')

    # 计算安全的刻度间隔，避免超过matplotlib的1000刻度限制
    max_ticks = 1000
    safe_gap = max(gap, data_len // max_ticks)

    if data_len // safe_gap <= max_ticks:
        ax.xaxis.set_major_locator(MultipleLocator(safe_gap + 1))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=max_ticks, prune='both'))

def score_plot(save_path: str,
               gap: float,
               test_data: np.ndarray,
               train_data: np.ndarray,
               threshold: float = None,
               figure_length: int = None,
               figure_width: int = None,
               font_size: int = None,
               recon_test_data: np.ndarray = None,
               recon_train_data: np.ndarray = None,
               test_anomaly_score: np.ndarray = None,
               train_anomaly_score: np.ndarray = None,
               test_label: np.ndarray = None,
               train_label: np.ndarray = None,
               plot_diff: bool = False):
    """
    @type threshold: object
    @param font_size:
    @param figure_width:
    @param figure_length:
    @param gap: axis gap
    @param save_path: save path/ png file name
    @param test_data: sequence_length, num_channels
    @param recon_test_data: sequence_length, num_channels
    @param train_data: sequence_length, num_channels
    @param recon_train_data: sequence_length, num_channels
    @param test_anomaly_score: sequence_length, num_channels
    @param train_anomaly_score: sequence_length, num_channels
    @param test_label: sequence_length
    @param train_label: sequence_length
    @return: None
    """

    sequence_length, dim_size = test_data.shape[0] + train_data.shape[0], train_data.shape[1]
    if figure_length is None or figure_width is None:
        if sequence_length >= 2000 and dim_size >= 8:
            figure_length, figure_width = sequence_length // 100, int(dim_size * 5)
        else:
            figure_length, figure_width = 20, 15

    if font_size is None:
        font_size = min(figure_length, figure_width) // dim_size

    # add a dimension for anomaly_score plotting
    n_dim = dim_size + 1

    ratio = train_data.shape[0] / test_data.shape[0]
    fig, axs = plt.subplots(
        nrows=n_dim, ncols=2,
        sharey=False, sharex=False,
        figsize=(figure_length, figure_width),
        tight_layout=True,
        gridspec_kw={'width_ratios': [ratio, 1]}
    )

    train_anomaly_segments = None
    test_anomaly_segments = None

    if train_label is not None:
        train_anomaly_segments = get_segments(train_label)

    if test_label is not None:
        test_anomaly_segments = get_segments(test_label)

    diff_train = []
    diff_test = []

    for dim in tqdm.trange(1, dim_size + 1, desc="plotting dim", unit="dim"):
        dim_train = train_data[:, dim - 1]
        dim_test = test_data[:, dim - 1]
        dim_train_anomaly_score = train_anomaly_score[:, dim - 1]
        dim_test_anomaly_score = test_anomaly_score[:, dim - 1]

        # plot train data
        axs[dim - 1][0].text(0.5, 0.8, f"train_data {dim}", fontsize=font_size)
        axs[dim - 1][0].plot(dim_train, label="raw")
        axs[dim - 1][0].plot(dim_train_anomaly_score, label="anomaly_score", color='black', linestyle='--')

        if recon_train_data is not None:
            axs[dim - 1][0].plot(recon_train_data[:, dim - 1], label="recon")
            diff_train_dim = abs(dim_train - recon_train_data[:, dim - 1]) * 0.8 - 0.2
            diff_train.append(diff_train_dim)
            if plot_diff:
                axs[dim - 1][0].plot(diff_train_dim, label="diff")

        # plot train label
        if train_anomaly_segments is not None:
            for seg in train_anomaly_segments:
                if seg[0] == seg[1]:
                    axs[dim - 1][0].axvline(x=seg[0], color='red', alpha=0.5)
                else:
                    axs[dim - 1][0].axvspan(seg[0], seg[1], facecolor='red', alpha=0.5)

        axs[dim - 1][0].legend(fontsize=font_size)
        axs[dim - 1][0].xaxis.set_major_locator(MultipleLocator(gap))
        axs[dim - 1][0].set_xlim(0, train_data.shape[0])

        # plot test data
        axs[dim - 1][1].text(0.5, 0.8, f"test_data {dim}", fontsize=font_size)
        axs[dim - 1][1].plot(dim_test, label="raw")
        axs[dim - 1][1].plot(dim_test_anomaly_score, label="anomaly", color='black', linestyle='--')

        if recon_test_data is not None:
            axs[dim - 1][1].plot(recon_test_data[:, dim - 1], label="recon")
            diff_test_dim = abs(dim_test - recon_test_data[:, dim - 1]) * 0.8 - 0.2
            diff_test.append(diff_test_dim)
            if plot_diff:
                axs[dim - 1][1].plot(diff_test_dim, label="diff")

        if test_anomaly_segments is not None:
            for seg in test_anomaly_segments:
                if seg[0] == seg[1]:
                    axs[dim - 1][1].axvline(x=seg[0], color='red', alpha=0.5)
                else:
                    axs[dim - 1][1].axvspan(seg[0], seg[1], facecolor='red', alpha=0.5)

        axs[dim - 1][1].legend(fontsize=font_size)
        axs[dim - 1][1].xaxis.set_major_locator(MultipleLocator(gap))
        axs[dim - 1][1].set_xlim(0, test_data.shape[0])

        y_max, y_min = (max(dim_train.max(), dim_test.max(), 1) + 0.1,
                        min(dim_train.min(), dim_test.min(), 0) - 0.2)

        axs[dim - 1][0].set_ylim(y_min, y_max)
        axs[dim - 1][1].set_ylim(y_min, y_max)

    # plot train anomaly score
    if train_anomaly_score is None and recon_train_data is not None:
        train_anomaly_score = np.array(diff_train).mean(0)
        # train_anomaly_score = (train_anomaly_score - train_anomaly_score.min()) / (
        #         train_anomaly_score.max() - train_anomaly_score.min())

    if train_anomaly_score is not None:
        axs[-1][0].plot(train_anomaly_score.mean(-1))

    if train_anomaly_segments is not None:
        for seg in train_anomaly_segments:
            if seg[0] == seg[1]:
                axs[-1][0].axvline(x=seg[0], color='red', alpha=0.3)
            else:
                axs[-1][0].axvspan(seg[0], seg[1], facecolor='red', alpha=0.3)

    axs[-1][0].xaxis.set_major_locator(MultipleLocator(gap))
    axs[-1][0].set_xlim(0, train_data.shape[0])

    # plot test anomaly score
    if test_anomaly_score is None and recon_test_data is not None:
        test_anomaly_score = np.array(diff_test).mean(0)
        # test_anomaly_score = (test_anomaly_score - test_anomaly_score.min()) / (
        #         test_anomaly_score.max() - test_anomaly_score.min())

    if test_anomaly_score is not None:
        axs[-1][1].plot(test_anomaly_score.mean(-1))

    if test_anomaly_segments is not None:
        for seg in test_anomaly_segments:
            if seg[0] == seg[1]:
                axs[-1][1].axvline(x=seg[0], color='red', alpha=0.3)
            else:
                axs[-1][1].axvspan(seg[0], seg[1], facecolor='red', alpha=0.3)

    if threshold is not None:
        axs[-1][1].axhline(y=threshold, color='red', alpha=0.5)

    axs[-1][1].xaxis.set_major_locator(MultipleLocator(gap))
    axs[-1][1].set_xlim(0, test_data.shape[0])

    plt.savefig(save_path, format="png", dpi=50)
    plt.close()

def tsne_plot(initial_feature_dict: Dict[str, np.ndarray],
              residual_feature_dict: Dict[str, np.ndarray],
              test_label_dict: Dict[str, np.ndarray],
              use_points_num: int = 100,
              save_dir: str = './figures',
              save_type: str = 'svg'):
    """
    color represent different data_name, shape represent label
    :param initial_feature_dict: [batch_size, n_patches, hidden_dim]
    :param residual_feature_dict: [batch_size, n_patches, hidden_dim]
    :param test_label_dict: [batch_size, n_patches]
    :param use_points_num: 每个数据集使用的点数
    :param save_dir: './figures'
    :param save_type: svg or pdf
    :return:
    """
    # 定义颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, len(initial_feature_dict)))
    color_map = {name: colors[i] for i, name in enumerate(initial_feature_dict.keys())}

    # 定义形状映射 (0: 圆形, 1: 三角形)
    shape_map = {0: 'o', 1: '^'}
    name_map = {0: 'normal', 1: 'anomaly'}

    # 为标签定义专用颜色（用于边缘分布图）
    label_colors = {0: '#2E86AB', 1: '#A23B72'}  # 蓝色表示normal，红色表示anomaly

    # 存储所有数据
    all_initial_features = []
    all_residual_features = []
    all_labels = []
    all_data_names = []

    # 合并所有数据
    for data_name in initial_feature_dict.keys():
        # 获取当前数据
        current_initial_feature = initial_feature_dict[data_name]
        current_residual_feature = residual_feature_dict[data_name]
        current_label = test_label_dict[data_name]

        batch_size, n_patches, hidden_dim = current_initial_feature.shape

        # 重塑数据
        current_initial_feature = current_initial_feature.reshape(batch_size * n_patches, hidden_dim)
        current_residual_feature = current_residual_feature.reshape(batch_size * n_patches, hidden_dim)
        current_label = current_label.reshape(batch_size * n_patches)

        # 采样数据点: 标签0采样use_points_num个,标签1全部保留
        label_normal_mask = current_label == 0
        label_anomaly_mask = current_label == 1

        label_normal_indices = np.where(label_normal_mask)[0]
        label_anomaly_indices = np.where(label_anomaly_mask)[0]

        # 对标签0进行采样
        if len(label_normal_indices) > use_points_num:
            sampled_label_normal_indices = np.random.choice(label_normal_indices, use_points_num, replace=False)
        else:
            sampled_label_normal_indices = label_normal_indices

        # 合并标签0的采样索引和标签1的全部索引
        selected_indices = np.concatenate([sampled_label_normal_indices, label_anomaly_indices])

        current_initial_feature = current_initial_feature[selected_indices]
        current_residual_feature = current_residual_feature[selected_indices]
        current_label = current_label[selected_indices]

        # 添加到列表
        all_initial_features.append(current_initial_feature)
        all_residual_features.append(current_residual_feature)
        all_labels.append(current_label)
        all_data_names.extend([data_name] * len(current_label))

    # 合并所有数据
    all_initial_features = np.vstack(all_initial_features)
    all_residual_features = np.vstack(all_residual_features)
    all_labels = np.concatenate(all_labels)
    all_data_names = np.array(all_data_names)

    # 对合并后的数据进行TSNE降维
    print("Performing TSNE on initial features...")
    tsne_initial = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_initial_features) - 1))
    initial_2d = tsne_initial.fit_transform(all_initial_features)

    print("Performing TSNE on residual features...")
    tsne_residual = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_residual_features) - 1))
    residual_2d = tsne_residual.fit_transform(all_residual_features)

    # 使用GridSpec创建带边缘分布图的布局
    fig = plt.figure(figsize=(27, 9))
    gs = GridSpec(2, 4, figure=fig,
                  height_ratios=[1, 8],  # 上方分布图:主图 = 1:6
                  width_ratios=[8, 1, 8, 1],  # 主图:右侧分布图 = 6:1
                  hspace=0.05, wspace=0.05)

    # 创建子图
    # 第一组：Initial Feature
    ax_initial_top = fig.add_subplot(gs[0, 0])  # 上方分布图
    ax_initial_main = fig.add_subplot(gs[1, 0])  # 主散点图
    ax_initial_right = fig.add_subplot(gs[1, 1])  # 右侧分布图

    # 第二组：Residual Feature
    ax_residual_top = fig.add_subplot(gs[0, 2])  # 上方分布图
    ax_residual_main = fig.add_subplot(gs[1, 2])  # 主散点图
    ax_residual_right = fig.add_subplot(gs[1, 3])  # 右侧分布图

    # ==================== 绘制Initial Feature ====================
    # 改动点：绘制主散点图（不添加图例
    for data_name in initial_feature_dict.keys():
        for label in [0, 1]:
            mask = (all_data_names == data_name) & (all_labels == label)
            ax_initial_main.scatter(initial_2d[mask, 0], initial_2d[mask, 1],
                                    c=[color_map[data_name]],
                                    marker=shape_map[label],
                                    alpha=0.6, s=50)

    # 绘制上方的X轴分布图（标签0和标签1）
    for label in [0, 1]:
        label_mask = all_labels == label
        data_x = initial_2d[label_mask, 0]

        # 使用KDE绘制平滑的分布曲线
        if len(data_x) > 1:
            kde = stats.gaussian_kde(data_x)
            x_range = np.linspace(data_x.min(), data_x.max(), 200)
            density = kde(x_range)
            ax_initial_top.fill_between(x_range, density, alpha=0.5,
                                        color=label_colors[label],
                                        label=name_map[label])
            ax_initial_top.plot(x_range, density, color=label_colors[label], linewidth=2)

    # 改动点：绘制右侧的Y轴分布图（标签0和标签1）
    for label in [0, 1]:
        label_mask = all_labels == label
        data_y = initial_2d[label_mask, 1]

        # 使用KDE绘制平滑的分布曲线
        if len(data_y) > 1:
            kde = stats.gaussian_kde(data_y)
            y_range = np.linspace(data_y.min(), data_y.max(), 200)
            density = kde(y_range)
            ax_initial_right.fill_betweenx(y_range, density, alpha=0.5,
                                           color=label_colors[label])
            ax_initial_right.plot(density, y_range, color=label_colors[label], linewidth=2)

    # calculate entropy for initial feature of label 0
    initial_2d_0 = initial_2d[all_labels == 0]
    initial_M = estimate_entropy_knn(initial_2d_0)
    # initial_2d_1 = initial_2d[all_labels == 1]
    # initial_M = compute_wasserstein_distance(initial_2d_0, initial_2d_1)


    # ==================== 绘制Residual Feature ====================
    # 绘制主散点图（不添加图例）
    for data_name in initial_feature_dict.keys():
        for label in [0, 1]:
            mask = (all_data_names == data_name) & (all_labels == label)
            ax_residual_main.scatter(residual_2d[mask, 0], residual_2d[mask, 1],
                                     c=[color_map[data_name]],
                                     marker=shape_map[label],
                                     alpha=0.6, s=50)

    # 改动点：绘制上方的X轴分布图（标签0和标签1）
    for label in [0, 1]:
        label_mask = all_labels == label
        data_x = residual_2d[label_mask, 0]

        if len(data_x) > 1:
            kde = stats.gaussian_kde(data_x)
            x_range = np.linspace(data_x.min(), data_x.max(), 200)
            density = kde(x_range)
            ax_residual_top.fill_between(x_range, density, alpha=0.5,
                                         color=label_colors[label],
                                         label=name_map[label])
            ax_residual_top.plot(x_range, density, color=label_colors[label], linewidth=2)

    # 改动点：绘制右侧的Y轴分布图（标签0和标签1）
    for label in [0, 1]:
        label_mask = all_labels == label
        data_y = residual_2d[label_mask, 1]

        if len(data_y) > 1:
            kde = stats.gaussian_kde(data_y)
            y_range = np.linspace(data_y.min(), data_y.max(), 200)
            density = kde(y_range)
            ax_residual_right.fill_betweenx(y_range, density, alpha=0.5,
                                            color=label_colors[label])
            ax_residual_right.plot(density, y_range, color=label_colors[label], linewidth=2)

    # calculate entropy for residual feature of label 0
    residual_2d_0 = residual_2d[all_labels == 0]
    # residual_2d_1 = residual_2d[all_labels == 1]
    residual_M = estimate_entropy_knn(residual_2d_0)

    # ==================== 设置图表属性 ====================
    # Initial Feature主图
    # ax_initial_main.set_title('Initial Feature TSNE', fontsize=14, fontweight='bold', pad=10)
    ax_initial_main.set_xlabel('TSNE Component 1', fontsize=11)
    ax_initial_main.set_ylabel('TSNE Component 2', fontsize=11)
    ax_initial_main.grid(True, alpha=0.3)

    # 为Initial主图添加简化的图例（仅显示标签）
    legend_elements_initial = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=10, label='Normal', alpha=0.6),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
               markersize=10, label='Anomaly', alpha=0.6)
    ]
    ax_initial_main.legend(handles=legend_elements_initial, loc='best', fontsize=10)

    # Residual Feature主图
    # ax_residual_main.set_title('Residual Feature TSNE', fontsize=14, fontweight='bold', pad=10)
    ax_residual_main.set_xlabel('TSNE Component 1', fontsize=11)
    # ax_residual_main.set_ylabel('TSNE Component 2', fontsize=11)
    ax_residual_main.grid(True, alpha=0.3)

    # 为Residual主图添加简化的图例（仅显示标签）
    legend_elements_residual = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=10, label='Normal', alpha=0.6),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
               markersize=10, label='Anomaly', alpha=0.6)
    ]
    ax_residual_main.legend(handles=legend_elements_residual, loc='best', fontsize=10)

    # 改动点：配置边缘分布图的样式
    # Initial上方分布图
    ax_initial_top.set_title(f'Initial Feature TSNE with Wasserstein {initial_M:.4e}', fontsize=18, fontweight='bold', pad=10)
    ax_initial_top.set_xlim(ax_initial_main.get_xlim())
    ax_initial_top.legend(loc='upper right', fontsize=8)
    ax_initial_top.tick_params(labelbottom=False)
    ax_initial_top.grid(True, alpha=0.3, axis='x')

    # Initial右侧分布图
    ax_initial_right.set_ylim(ax_initial_main.get_ylim())
    ax_initial_right.tick_params(labelleft=False)
    ax_initial_right.grid(True, alpha=0.3, axis='y')

    # Residual上方分布图
    ax_residual_top.set_title(f'Residual Feature TSNE with Wasserstein {residual_M:.4e}', fontsize=18, fontweight='bold', pad=10)
    ax_residual_top.set_xlim(ax_residual_main.get_xlim())
    ax_residual_top.legend(loc='upper right', fontsize=8)
    ax_residual_top.tick_params(labelbottom=False)
    ax_residual_top.grid(True, alpha=0.3, axis='x')

    # Residual右侧分布图
    ax_residual_right.set_ylim(ax_residual_main.get_ylim())
    ax_residual_right.tick_params(labelleft=False)
    ax_residual_right.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if save_type == "svg":
        plt.savefig(os.path.join(save_dir, "tsne.svg"), bbox_inches='tight', dpi=300)
    elif save_type == "pdf":
        plt.savefig(os.path.join(save_dir, "tsne.pdf"), bbox_inches='tight', dpi=300)
    else:
        plt.savefig(os.path.join(save_dir, "tsne.png"), bbox_inches='tight', dpi=300)

    # plt.show()

    return fig


if __name__ == "__main__":
    segments = get_segments(np.array([1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]))
    print(segments)
    # raw_train_data, raw_test_data, labels = load_dataset(data_name="UCR", group="235")
    # recon_plot(save_path="sample.png", train_data=raw_train_data, test_data=raw_test_data, test_label=labels, gap=400)
    # print(segments)
    # print("finished")
