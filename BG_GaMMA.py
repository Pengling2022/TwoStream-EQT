import glob
import os
import importlib

import matplotlib
matplotlib.use('Agg')


import gc
# import psutil
from IPython.display import clear_output
# 指定新的缓存路径
# new_cache_path = "/scratch/groups/beroza/lei/seisbench-test/cache"
#
# # 设置环境变量
# os.environ["SEISBENCH_CACHE_ROOT"] = new_cache_path

# 导入 seisbench 并验证新的缓存路径是否生效
import seisbench
importlib.reload(seisbench)

# print(f"New cache path from environment variable: {seisbench.cache_root}")


# import seisbench
from obspy.clients.fdsn import Client
import obspy
import matplotlib.pyplot as plt
import obspy as obs
import numpy as np
from collections import Counter
from tqdm import tqdm
import seaborn as sns
import torch
import seisbench.models as sbm
import pandas as pd
from obspy import read, Stream, UTCDateTime
from obspy import UTCDateTime
from pyproj import CRS, Transformer
#
#
from gamma.utils import association
import seisbench.models as sbm
sns.set(font_scale=1.2)
sns.set_style("ticks")

# 共用台站信息和速度模型
client = Client("NCEDC")
t0 = UTCDateTime("2023/01/01 00:00:00")
t1 = t0 + 24 * 60 * 60
# t1 = t0 + 24 * 60 * 60   # Full day, requires more memory
# stream = client.get_waveforms(network="CX", station="*", location="*", channel="HH?", starttime=t0, endtime=t1)

inv = client.get_stations(network="BG", station="*", location="*", channel="*", starttime=t0, endtime=t1)
# print(inv)

# Projections
wgs84 = CRS.from_epsg(4326)
local_crs = CRS.from_epsg(26910)  # NAD83 / UTM zone 10N, 适用于北加州西部
# 如果你的数据更靠东，可以使用 26911 (NAD83 / UTM zone 11N)
transformer = Transformer.from_crs(wgs84, local_crs)
station_df = []
for station in inv[0]:
    station_df.append({
        "id": f"BG.{station.code}.",
        "longitude": station.longitude,
        "latitude": station.latitude,
        "elevation(m)": station.elevation
    })
station_df = pd.DataFrame(station_df)
# print(station_df)

station_df["x(km)"] = station_df.apply(lambda x: transformer.transform(x["latitude"], x["longitude"])[0] / 1e3, axis=1)
station_df["y(km)"] = station_df.apply(lambda x: transformer.transform(x["latitude"], x["longitude"])[1] / 1e3, axis=1)
station_df["z(km)"] = -station_df["elevation(m)"] / 1e3

# print(station_df["x(km)"])
# print(station_df["y(km)"])
# print(station_df["z(km)"])

northing = {station: y for station, y in zip(station_df["id"], station_df["y(km)"])}
station_dict = {station: (x, y) for station, x, y in zip(station_df["id"], station_df["x(km)"], station_df["y(km)"])}


## GaMMA参数配置
config = {}
config["dims"] = ['x(km)', 'y(km)', 'z(km)']
config["use_dbscan"] = True
config["use_amplitude"] = False #如何利用振幅信息？
config["x(km)"] = (500, 530)
config["y(km)"] = (4280, 4310)
config["z(km)"] = (0, 10)

# 示例层状速度模型  层状速度模型报错？
# 每个层的深度边界 (km)
# depths = [0, 1.5, 3.0, 4.25, 6.0, 8.0]
# # 对应的 P 波速度 (km/s)
# vp = [4.43, 5.12, 5.47, 5.58, 5.62, 5.86]
# # 对应的 S 波速度 (km/s)
# vp_vs_ratio = 1.73
# vs = [v / vp_vs_ratio for v in vp]

# # 定义速度模型字典
# velocity_model = {
#     "z": depths,
#     "p": vp,
#     "s": vs
# }
# config["vel"] = velocity_model  # Velocity model

config["vel"] = {"p": 5.2, "s": 3.0}  # Velocity model (homogeneous)
config["method"] = "BGMM"
if config["method"] == "BGMM":
    config["oversample_factor"] = 4
if config["method"] == "GMM":
    config["oversample_factor"] = 1

# DBSCAN
config["bfgs_bounds"] = (
    (config["x(km)"][0] - 1, config["x(km)"][1] + 1),  # x
    (config["y(km)"][0] - 1, config["y(km)"][1] + 1),  # y
    (0, config["z(km)"][1] + 1),  # x
    (None, None),  # t
)
config["dbscan_eps"] = 5  # seconds
config["dbscan_min_samples"] = 20

# Filtering
config["min_picks_per_eq"] = 24 # Minimum picks for associated earthquakes
config["min_p_picks_per_eq"] = 4 #
config["max_sigma11"] = 0.3 # Max phase time residual (s)
config["max_sigma22"] = 1.0
config["max_sigma12"] = 1.0


station_df = []
for station in inv[0]:
    station_df.append({
        "id": f"BG.{station.code}.",
        "longitude": station.longitude,
        "latitude": station.latitude,
        "elevation(m)": station.elevation
    })
station_df = pd.DataFrame(station_df)
# print(station_df)

station_df["x(km)"] = station_df.apply(lambda x: transformer.transform(x["latitude"], x["longitude"])[0] / 1e3, axis=1)
station_df["y(km)"] = station_df.apply(lambda x: transformer.transform(x["latitude"], x["longitude"])[1] / 1e3, axis=1)
station_df["z(km)"] = station_df["elevation(m)"] / 1e3

# print(station_df["x(km)"])
# print(station_df["y(km)"])
# print(station_df["z(km)"])
if __name__ == '__main__':
    northing = {station: y for station, y in zip(station_df["id"], station_df["y(km)"])}
    station_dict = {station: (x, y) for station, x, y in zip(station_df["id"], station_df["x(km)"], station_df["y(km)"])}

    # 指定路径
    folder_path = r"G:/EQT"
    # 构建通配符路径
    csv_files = glob.glob(os.path.join(folder_path, "EQT_pick_original*_task23.csv"))

    # 依次读取所有符合条件的CSV文件
    for i, file in enumerate(csv_files):
        pick_df = pd.read_csv(file)
        print(f"读取文件: {file}")
        #
        catalogs, assignments = association(pick_df, station_df, config, method=config["method"])

        catalog = pd.DataFrame(catalogs)
        assignments = pd.DataFrame(assignments, columns=["pick_idx", "event_idx", "prob_gamma"])

        catalog.to_csv(f"{folder_path}/catalog{i}.csv")
        assignments.to_csv(f"{folder_path}/assignments{i}.csv")

#     process = psutil.Process(os.getpid())
#     print(f"Memory usage after processing {subfolders[num]}: {process.memory_info().rss / (1024 ** 2)} MB")






