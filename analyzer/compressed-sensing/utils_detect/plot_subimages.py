import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as font_manager


def plot_raw_kpi(raw, column_names=None, true=[], shape=30, title=[], save_path=None):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 16

    ny, nx = raw.shape
    f, ax = plt.subplots(nx, 1, sharex='all', figsize=(shape, shape))

    # f.suptitle('Raw KPI Stream', verticalalignment='center')
    if nx == 1:
        ax = [ax]

    for i in range(nx):
        kpi_name = column_names[i] if column_names and i < len(column_names) else title[i] if i < len(title) else ""
        ax[i].plot(range(ny), raw[:, i], label=kpi_name)
        ax[i].set_xlim([0, ny])
        # 如果有列名，用列名替代title
        if column_names and i < len(column_names):
            ax[i].text(0.0, 0.75, str(column_names[i]), fontfamily='Times New Roman', weight='heavy', 
                      bbox=dict(boxstyle="square",edgecolor='#e4e4e4', facecolor="white", alpha=0.8), zorder=20)
        elif len(title) > 0:
            ax[i].text(0.0, 0.75, str(title[i]), fontfamily='Times New Roman', weight='heavy',
                      bbox=dict(boxstyle="square",edgecolor='#e4e4e4', facecolor="white", alpha=0.8), zorder=20)
        
        if len(true) > 0:
            ax[i].vlines(true, 0, 1, colors='#f4a09c',
                         linewidth=shape/3, alpha=0.8)
            # ax[i].add_patch(patches.Rectangle((true[0], 0.1),0.5,0.5))
        ax[i].set_yticks([])
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
    # ax[0].legend(loc=9, bbox_to_anchor=(0.5, 1.23), ncol=2)
    ax[nx-1].spines['bottom'].set_visible(True)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()


def plot_cluster_kpi(raw, cluster,column_names=None, true=[], shape=30, title=[], save_path=None):

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 16
    color_map = ['#ffffff', '#f7f7f7', '#fae6d1', '#faefd1', '#faf9d1', '#ebfad1', '#d1fad7', '#d1faf0', '#d1f3fa', '#d1e3fa', '#d4d1fa', '#e5d1fa', '#fad1e6', '#fad3d1', '#faddd1'] # #f7f7f7 灰色，白灰相间

    ny, nx = raw.shape
    f, ax = plt.subplots(nx, 1, sharex='all', figsize=(shape, shape))

    if nx == 1:
        ax = [ax]

    i = 0
    while i < nx:
        for index in range(len(cluster)):
            for item in cluster[index]:
                ax[i].patch.set_facecolor(color_map[index%2])
                ax[i].plot(range(ny), raw[:, item], label='Raw KPI')
                ax[i].set_xlim([0, ny])
                ax[i].set_ylim([0, 1])
                # if len(title) > 0:
                #     ax[i].text(0.0, 0.75, str(title[item]), fontfamily='Times New Roman', weight='heavy', bbox = dict(boxstyle="square",edgecolor='#e4e4e4', facecolor='w', alpha=0.6), zorder=20)
                # 如果有列名，用列名替代title
                if column_names and i < len(column_names):
                    ax[i].text(0.0, 0.75, str(column_names[i]), fontfamily='Times New Roman', weight='heavy', 
                            bbox=dict(boxstyle="square",edgecolor='#e4e4e4', facecolor="white", alpha=0.8), zorder=20)
                elif len(title) > 0:
                    ax[i].text(0.0, 0.75, str(title[item]), fontfamily='Times New Roman', weight='heavy',
                            bbox=dict(boxstyle="square",edgecolor='#e4e4e4', facecolor="white", alpha=0.8), zorder=20)
        
                if len(true) > 0:
                    ax[i].vlines(true, 0, 1, colors='#f4a09c', linewidth=shape/3, alpha=0.8)
                    # ax[i].add_patch(patches.Rectangle((true[0], 0.1),0.5,0.5))
                ax[i].set_yticks([])
                # 
                ax[i].spines['top'].set_visible(False)
                ax[i].spines['right'].set_visible(False)
                ax[i].spines['bottom'].set_visible(False)
                ax[i].spines['left'].set_visible(False)
                i = i + 1
    ax[0].legend(loc=9, bbox_to_anchor=(0.5, 1.23), ncol=2)
    ax[nx-1].spines['bottom'].set_visible(True)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()


def plot_sample_kpi(raw,  sample_point, cluster, true=[], column_names=None, shape: int = 30, title=[], save_path=None):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 16

    ny, nx = raw.shape
    f, ax = plt.subplots(nx, 1, sharex='all', figsize=(shape, shape))
    item_array = []
    for item in cluster:
        item_array = item_array + item

    if nx == 1:
        ax = [ax]
    i = 0
    while i < nx:
        ax[i].plot(range(ny), raw[:, item_array[i]], label='Raw KPI')
        ax[i].scatter(sample_point, raw[sample_point, item_array[i]],
                      marker='o', c='r', zorder=20, label='Sample Point', alpha=0.5)
        ax[i].set_xlim([0, ny])
        ax[i].set_ylim([0, 1])
        # if len(title) > 0:
        #     ax[i].text(0.0, 0.75, str(title[i]), fontfamily='Times New Roman', weight='heavy', bbox = dict(boxstyle="square",edgecolor='#e4e4e4', facecolor="white", alpha=0.8), zorder=20)
        # 如果有列名，用列名替代title
        if column_names and i < len(column_names):
            ax[i].text(0.0, 0.75, str(column_names[i]), fontfamily='Times New Roman', weight='heavy', 
                      bbox=dict(boxstyle="square",edgecolor='#e4e4e4', facecolor="white", alpha=0.8), zorder=20)
        elif len(title) > 0:
            ax[i].text(0.0, 0.75, str(title[i]), fontfamily='Times New Roman', weight='heavy',
                      bbox=dict(boxstyle="square",edgecolor='#e4e4e4', facecolor="white", alpha=0.8), zorder=20)
        
        if len(true) > 0:
            ax[i].vlines(true, 0, 1, colors='#f4a09c',
                         linewidth=shape/3, alpha=0.8)
            # ax[i].add_patch(patches.Rectangle((true[0], 0.1),0.5,0.5))
        ax[i].set_yticks([])
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        i = i + 1
    ax[0].legend(loc=9, bbox_to_anchor=(0.5, 1.23), ncol=2)
    ax[nx-1].spines['bottom'].set_visible(True)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()


def plot_reconstruct_kpi(raw, reconstruct,column_names=None,  true=[], shape: int = 30, title=[], save_path=None):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 16

    ny, nx = raw.shape
    f, ax = plt.subplots(nx, 1, sharex='all', figsize=(shape, shape))
    if nx == 1:
        ax = [ax]
    i = 0
    while i < nx:
        # ax[i].plot(range(ny), raw[:, i], label='Raw KPI')
        # ax[i].plot(range(ny), reconstruct[:, i], label='Reconstruct KPI', c='#ff8f65')
        kpi_name = column_names[i] if column_names and i < len(column_names) else f'Dimension {i+1}'
        
        ax[i].plot(range(ny), raw[:, i], label=f'(Raw)')
        ax[i].plot(range(ny), reconstruct[:, i], label=f'(Reconstructed)', c='#ff8f65')
        ax[i].set_xlim([0, ny])
        ax[i].set_ylim([0, 1])
        # if len(title) > 0:
        #     ax[i].text(0.0, 0.75, str(title[i]), fontfamily='Times New Roman', weight='heavy', bbox = dict(boxstyle="square",edgecolor='#e4e4e4', facecolor="white", alpha=0.8), zorder=20)
        # 如果有列名，用列名替代title
        if column_names and i < len(column_names):
            ax[i].text(0.0, 0.75, str(column_names[i]), fontfamily='Times New Roman', weight='heavy', 
                      bbox=dict(boxstyle="square",edgecolor='#e4e4e4', facecolor="white", alpha=0.8), zorder=20)
        elif len(title) > 0:
            ax[i].text(0.0, 0.75, str(title[i]), fontfamily='Times New Roman', weight='heavy',
                      bbox=dict(boxstyle="square",edgecolor='#e4e4e4', facecolor="white", alpha=0.8), zorder=20)
        
        if len(true) > 0:
            ax[i].vlines(true, 0, 1, colors='#f4a09c',
                         linewidth=shape/3, alpha=0.8)
            # ax[i].add_patch(patches.Rectangle((true[0], 0.1),0.5,0.5))
        ax[i].set_yticks([])
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        i = i + 1
    ax[0].legend(loc=9, bbox_to_anchor=(0.5, 1.23), ncol=2)
    ax[nx-1].spines['bottom'].set_visible(True)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()
