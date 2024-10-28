import gc

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from processing_data.processing_diting import *

if __name__ == "__main__":
    # change path here
    DiTingDatasetPath = r'D:/diting/Diting50hz/'

    DiTing_330km_csv = pd.read_csv(DiTingDatasetPath + 'DiTing330km_total.csv', dtype={'key': str}, low_memory=False)
    DiTing_330km_csv['p_pick'] = DiTing_330km_csv['p_pick'].astype(float)
    DiTing_330km_csv['s_pick'] = DiTing_330km_csv['s_pick'].astype(float)

    start_index = 0
    chunk_size = 100000
    # 生成索引列表，从指定的start_index开始
    index_list = list(range(start_index, len(DiTing_330km_csv), chunk_size))

    for chunk_num, start_idx in enumerate(index_list[20:]):
        end_idx = start_idx + chunk_size
        if end_idx > len(DiTing_330km_csv):
            end_idx = len(DiTing_330km_csv)
        # print(start_idx, end_idx)
        # print(start_idx//chunk_size)

        sub_csv = DiTing_330km_csv.iloc[start_idx:end_idx].copy()
        # 添加清洗标签列flag
        clear = True
        sub_csv['flag'] = False
        sub_csv['cut_start'] = np.nan
        sub_csv['p_t'] = np.nan
        sub_csv['s_t'] = np.nan

        for row_idx in tqdm(range(len(sub_csv))):
            p = np.nan
            s = np.nan
            row_index = sub_csv.index[row_idx]
            plt.close('all')

            fig = plt.figure(figsize=(15, 10))
            axs = fig.subplots(3, 1, sharex=True, gridspec_kw={'hspace': 0, 'height_ratios': (1, 1, 1)})

            part = int(sub_csv.iloc[row_idx]['part'])
            key = sub_csv.iloc[row_idx]['key']
            key_correct = key.split('.')
            key1 = key_correct[0].rjust(6, '0') + '.' + key_correct[1].ljust(4, '0')

            p_t = sub_csv.iloc[row_idx]['p_pick']
            s_t = sub_csv.iloc[row_idx]['s_pick']
            # 获取数据

            data = get_from_DiTing(part=part, key=key1, h5file_path=DiTingDatasetPath)
            # # 裁剪数据
            cropped_data, new_p_index, new_s_index,cut_start= crop_dataset(data, p_t, s_t,length=3000)

            sub_csv.loc[row_index, 'p_pick'] = new_p_index
            sub_csv.loc[row_index, 's_pick'] = new_s_index
            sub_csv.loc[row_index, 'cut_start'] = cut_start

        # 清洗数据
            if clear:
                flag,p,s=predict_DiTing(cropped_data)
                # 更改csv
                if flag:
                    if p and abs(new_p_index - p) < 100:
                        sub_csv.at[row_index, "flag"] = True
                        sub_csv.at[row_index, "p_t"] = p
                    if s and abs(new_s_index - s) < 100:
                        sub_csv.at[row_index, "s_t"] = s

            file_path = f'G:/Diting/waveforms_{start_idx//chunk_size}.hdf5'
            # file_path = f'waveforms_{i}.hdf5'
            group_path = "earthquake"  # 组的路径
            dataset_key = key1  # 指定的键值，你可以根据需要修改
            stream = cropped_data

            write_to_new_hdf5(file_path, group_path, dataset_key, stream)

            gc.collect()
        sub_csv['part'] = int(start_idx//chunk_size)
        # sub_csv.to_csv(f'DiTing330km_part_{i}.csv', index=False)
        sub_csv.to_csv(f'G:/Diting/DiTing330km_part_{start_idx//chunk_size}.csv', index=False)

            # # 绘图
            # plt.close('all')
            # fig = plt.figure(figsize=(15, 10))
            # axs = fig.subplots(3, 1, sharex=True, gridspec_kw={'hspace': 0, 'height_ratios': (1, 1, 1)})
            #
            # axs[0].plot(cropped_data[0].data, linewidth=2, c='black', label='Z')  # 加粗线条
            # axs[0].axvline(x=new_p_index, color='r', label='Manual_P_Arrival')  # 在x=10处添加一条红色的虚线
            # axs[0].axvline(x=new_s_index, color='b', label='Manual_S_Arrival')  # 在x=10处添加一条红色的虚线
            # if p:
            #     axs[0].axvline(x=p, color='r', linestyle='--', label='Average_P_Arrival')  # 在x=10处添加一条红色的虚线
            # if s:
            #     axs[0].axvline(x=s, color='b', linestyle='--', label='Average_S_Arrival')  # 在x=10处添加一条红色的虚线
            # # axs[0].set_ylabel('Amplitude', fontweight='bold', fontsize=16)
            # # axs[0].set_ylim(-1.1, 1.1)
            #
            # axs[1].plot(cropped_data[1].data, linewidth=2, c='black', label='N')  # 加粗线条
            # axs[1].axvline(x=new_p_index, color='r', label='Manual_P_Arrival')  # 在x=10处添加一条红色的虚线
            # axs[1].axvline(x=new_s_index, color='b', label='Manual_S_Arrival')  # 在x=10处添加一条红色的虚线
            # if p:
            #     axs[1].axvline(x=p, color='r', linestyle='--', label='Average_P_Arrival')  # 在x=10处添加一条红色的虚线
            # if s:
            #     axs[1].axvline(x=s, color='b', linestyle='--', label='Average_S_Arrival')  # 在x=10处添加一条红色的虚线
            # # axs[1].set_ylabel('Amplitude', fontweight='bold', fontsize=16)
            # # axs[1].set_ylim(-1.1, 1.1)
            #
            # axs[2].plot(cropped_data[2].data, linewidth=2, c='black', label='E')  # 加粗线条
            # axs[2].axvline(x=new_p_index, color='r', label='Manual_P_Arrival')  # 在x=10处添加一条红色的虚线
            # axs[2].axvline(x=new_s_index, color='b', label='Manual_S_Arrival')  # 在x=10处添加一条红色的虚线
            # if p:
            #     axs[2].axvline(x=p, color='r', linestyle='--', label='Average_P_Arrival')  # 在x=10处添加一条红色的虚线
            # if s:
            #     axs[2].axvline(x=s, color='b', linestyle='--', label='Average_S_Arrival')  # 在x=10处添加一条红色的虚线
            # # axs[2].set_ylabel('Amplitude', fontweight='bold', fontsize=16)
            # # axs[2].set_ylim(-1.1, 1.1)
            #
            #
            # # 设置坐标轴加粗和调整字体大小
            # for ax in axs:
            #     ax.spines['top'].set_linewidth(2)
            #     ax.spines['right'].set_linewidth(2)
            #     ax.spines['bottom'].set_linewidth(2)
            #     ax.spines['left'].set_linewidth(2)
            #     ax.tick_params(axis='x', which='major', width=2, labelsize=18)  # 增加 x 轴刻度线的宽度
            #     ax.tick_params(axis='y', which='major', width=2, labelsize=16)  # 增加 y 轴刻度线的宽度
            #
            # axs[0].legend(loc='upper right', fontsize=14, frameon=False)  # 添加自定义的图例到第一个子图
            # axs[1].legend(loc='upper right', fontsize=14, frameon=False)  # 添加自定义的图例到第一个子图
            # axs[2].legend(loc='upper right', fontsize=14, frameon=False)  # 添加自定义的图例到第一个子图
            # plt.show()
