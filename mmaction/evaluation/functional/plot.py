import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp


### x轴是label; y轴是pred
def plot(x_data, y_data, save_path, save_name):
    # 设置背景颜色
    sns.set_style("white")

    # 设置x轴和y轴的最大值
    plt.xlim(-2, 45)  # 设置x轴的最大值为50
    plt.ylim(-2, 45)  # 设置y轴的最大值为45
    plt.rc('font', family='Times New Roman')
    # 使用regplot绘制散点图并计算回归线的斜率和截距
    # ci=None表示不显示置信区间
    # 灰色：#95a5a6
    # 蓝色：#3498db
    # 紫色：#9b59b6
    # 绿色：#28B279
    sns.regplot(x=x_data, y=y_data, ci=95, fit_reg=True,
                scatter_kws={"s": 60}, line_kws={'linewidth': 4})
                # scatter_kws={"color": "#0FEF9E", "alpha": 0.8, "s": 40})

    # # 获取当前图和轴
    ax = plt.gca()

    # 调整刻度大小
    ax.tick_params(axis='x', labelsize=13)  # 设置X轴刻度大小为20
    ax.tick_params(axis='y', labelsize=13)  # 设置Y轴刻度大小为20

    # 在散点图上画一条直线
    # ax.axline((10, 10), (20, 20), color='red')
    ax.plot((0, 43), (0, 43), color='red', linewidth=4.0)

    plt.savefig(osp.join(save_path, save_name), bbox_inches='tight', pad_inches=0.1)

    plt.close()