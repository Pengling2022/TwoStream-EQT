import numpy as np
import pandas as pd
import gc

from matplotlib import pyplot as plt
from tqdm import tqdm
from processing_data.processing_stead import read_stead, predict_stead
from obspy import Stream, Trace
import seisbench.data as sbd

if __name__ == "__main__":
    # Data load
    metadata = pd.read_csv("G:\\merged\\merged.csv",low_memory=False)
    # Initialize new columns
    metadata['flag'] = False
    metadata['p_t'] = np.nan
    metadata['s_t'] = np.nan

    hdf5_path = "G:\\merged\\merged.hdf5"

    # Separate noise data
    # noise_data = metadata[metadata["trace_category"] == "noise"]
    metadata = metadata[metadata["trace_category"] != "noise"].copy()  # Use .copy() to avoid SettingWithCopyWarning

    # # Save noise data
    # noise_output_file = 'G:\\noise_data.csv'
    # noise_data.to_csv(noise_output_file, index=False)

    start_index = 0
    chunk_size = 100000
    # 生成索引列表，从指定的start_index开始
    index_list = list(range(start_index, len(metadata), chunk_size))

    for chunk_num, start_idx in enumerate(index_list[6:7]):
        end_idx = start_idx + chunk_size
        if end_idx > len(metadata):
            end_idx = len(metadata)
        print(start_idx, end_idx)

        processed_chunk = metadata.iloc[start_idx:end_idx].copy()

        for i in tqdm(range(len(processed_chunk))):
            row_index = processed_chunk.index[i]
            trace_category = processed_chunk.iloc[i]["trace_category"]
            if trace_category == "noise":
                processed_chunk.at[i, 'flag'] = True
                continue  # Skip the rest of the loop for this iteration

            trace_name = processed_chunk.iloc[i]["trace_name"]
            p_t = processed_chunk.iloc[i]['p_arrival_sample']
            s_t = processed_chunk.iloc[i]['s_arrival_sample']

            Data = read_stead(hdf5_path, trace_name)
            stream = Stream(
                traces=[Trace(data=Data[:, j], header={"sampling_rate": 100, "channel": f"HH{'ZNE'[j]}"}) for j in
                        range(Data.shape[1])])

            # Prediction
            flag, p, s = predict_stead(stream)
            if flag:
                if p and abs(p_t - p) < 100:
                    processed_chunk.at[row_index, "flag"] = True
                    processed_chunk.at[row_index, "p_t"] = p
                if s and abs(s_t-s)<100:
                    processed_chunk.at[row_index, "s_t"] = s

        # Clean up memory
        gc.collect()

        # 保存处理后的块为单独的CSV文件
        output_file = f'G:\processed_chunk_{start_idx//chunk_size}.csv'
        # print(output_file)
        processed_chunk.to_csv(output_file, index=False)
        # 清理内存
        gc.collect()

            # plt.close('all')
            # fig = plt.figure(figsize=(15, 10))
            # axs = fig.subplots(3, 1, sharex=True, gridspec_kw={'hspace': 0, 'height_ratios': (1, 1, 1)})
            #
            # axs[0].plot(Data[:, 0], linewidth=2, c='black', label='Z')  # 加粗线条
            # axs[0].axvline(x=p_t, color='r', label='Manual_P_Arrival')  # 在x=10处添加一条红色的虚线
            # axs[0].axvline(x=s_t, color='b', label='Manual_S_Arrival')  # 在x=10处添加一条红色的虚线
            # if p:
            #     axs[0].axvline(x=p, color='r', linestyle='--', label='Average_P_Arrival')  # 在x=10处添加一条红色的虚线
            # if s:
            #     axs[0].axvline(x=s, color='b', linestyle='--', label='Average_S_Arrival')  # 在x=10处添加一条红色的虚线
            # # axs[0].set_ylabel('Amplitude', fontweight='bold', fontsize=16)
            # # axs[0].set_ylim(-1.1, 1.1)
            #
            # axs[1].plot(Data[:, 1], linewidth=2, c='black', label='N')  # 加粗线条
            # axs[1].axvline(x=p_t, color='r', label='Manual_P_Arrival')  # 在x=10处添加一条红色的虚线
            # axs[1].axvline(x=s_t, color='b', label='Manual_S_Arrival')  # 在x=10处添加一条红色的虚线
            # if p:
            #     axs[1].axvline(x=p, color='r', linestyle='--', label='Average_P_Arrival')  # 在x=10处添加一条红色的虚线
            # if s:
            #     axs[1].axvline(x=s, color='b', linestyle='--', label='Average_S_Arrival')  # 在x=10处添加一条红色的虚线
            # # axs[1].set_ylabel('Amplitude', fontweight='bold', fontsize=16)
            # # axs[1].set_ylim(-1.1, 1.1)
            #
            # axs[2].plot(Data[:, 2], linewidth=2, c='black', label='E')  # 加粗线条
            # axs[2].axvline(x=p_t, color='r', label='Manual_P_Arrival')  # 在x=10处添加一条红色的虚线
            # axs[2].axvline(x=s_t, color='b', label='Manual_S_Arrival')  # 在x=10处添加一条红色的虚线
            # if p:
            #     axs[2].axvline(x=p, color='r', linestyle='--', label='Average_P_Arrival')  # 在x=10处添加一条红色的虚线
            # if s:
            #     axs[2].axvline(x=s, color='b', linestyle='--', label='Average_S_Arrival')  # 在x=10处添加一条红色的虚线
            # # axs[2].set_ylabel('Amplitude', fontweight='bold', fontsize=16)
            # # axs[2].set_ylim(-1.1, 1.1)
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
            #
            # # plt.savefig('figure_s/EQT_{}_1.png'.format(idx), bbox_inches='tight')  # 保存图像时自动裁剪边缘空白部分
            # # plt.title(processed_chunk.iloc[idx]["flag"])
            # plt.show()
            #
            #
