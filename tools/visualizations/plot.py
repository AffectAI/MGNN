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
    plt.xlim(-10, 50)  # 设置x轴的最大值为50
    plt.ylim(-3, 45)  # 设置y轴的最大值为45

    # 使用regplot绘制散点图并计算回归线的斜率和截距
    sns.regplot(x=x_data, y=y_data, ci=None, fit_reg=False)  # ci=None表示不显示置信区间

    # # 获取当前图和轴
    ax = plt.gca()

    # 调整刻度大小
    ax.tick_params(axis='x', labelsize=20)  # 设置X轴刻度大小为20
    ax.tick_params(axis='y', labelsize=20)  # 设置Y轴刻度大小为20

    # # 在散点图上画一条直线
    ax.axline((0, 0), slope=1, color='red')

    plt.savefig(osp.join(save_path, save_name), bbox_inches='tight', pad_inches=0.1)

    plt.close()