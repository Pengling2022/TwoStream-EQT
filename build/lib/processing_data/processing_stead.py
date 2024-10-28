import gc
import time

import h5py
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from obspy import Stream, Trace
from scipy.signal import find_peaks
from scipy.stats import zscore
from tqdm import tqdm
import seisbench.models as sbm

import h5py
import torch.nn.functional as F

def read_stead(hdf5_path, trace_name):
    with h5py.File(hdf5_path, 'r') as f:
        # Access the 'data' group
        gdata = f["data"]
        # Use the trace_name from the metadata to extract the waveform from the HDF5 file
        x = gdata[trace_name][()]
        return x

def write_to_new_hdf5(file_path, group_path, dataset_key, stream):
    key3 = dataset_key
    data = np.array([trace.data for trace in stream]).T
    with h5py.File(file_path, 'a') as f:
        # 创建组
        if group_path in f:
            group = f[group_path]  # 如果组已经存在，则直接获取现有组
        else:
            group = f.create_group(group_path)
        # 创建数据集
        # group[dataset_key] = data
        if dataset_key in group:
            del group[dataset_key]  # 删除已有数据集
        group.create_dataset(dataset_key, data=data)

def find_ps_peaks(preds,st,p_th=0.5,s_th=0.3,dist=1.5):
    pick_dict={}
    height_dict={"P":p_th,"S":s_th}
    offset = preds[0].stats.starttime - st
    for pred_trace in preds:
        model, pred_class = pred_trace.stats.channel.split("_")
        if pred_class == "P" or pred_class == "S":
            pick,_=find_peaks(pred_trace.data,height=height_dict[pred_class],distance=dist*pred_trace.stats.sampling_rate)
            pick=pred_trace.times()[pick]+offset
            pick_dict[pred_class]=pick
    return pick_dict

def average_arrival(picks,type,p_win_s=0.15,s_win_s=0.2,z_threshold=2):
    # 删除多事件，没有拾取成功的
    time_dict={"P":p_win_s,"S":s_win_s}
    n=len(picks)
    output_dict={}
    output_dict["multievent"]=[]
    temp=[]
    ind=[]
    for i,pick in enumerate(picks):
        if len(pick)!=1:
            output_dict["multievent"].append(i)
            continue
        temp.append(pick[0])
        ind.append(i)

    # 删除离群点
    picks = np.array(temp)
    z_scores = zscore(picks)
    outliers = np.where(np.abs(z_scores) > z_threshold)
    output_dict["outliers"]=[ind[i] for i in outliers[0]]
    picks = np.delete(picks, outliers)

    # 判断
    if len(picks)>=np.floor(((n+1)/2)) and np.max(picks)-np.min(picks)<time_dict[type]:
        output_dict["arrival"]=DWG(picks)
        return True,output_dict
    else:
        return False,output_dict

def DWG(picks_num):
    weights = []
    for i in range(picks_num.shape[0]):
        weight=0.0
        for j in range(picks_num.shape[0]):
            if j!=i:
                weight=weight+np.abs(picks_num[i]-picks_num[j])
        if np.isclose(weight,0.0):
            return picks_num[0]
        weights.append(1/weight)
    sum=np.sum(np.array(weights))
    res=0.0
    for i in range(picks_num.shape[0]):
        res=res+weights[i]*picks_num[i]
    return res/sum

def predict_stead(stream):
    datasets = ["scedc", "stead"]  # ,'ethz',,'geofon'
    models = [sbm.OBSTransformer.from_pretrained("obst2024"), sbm.GPD.from_pretrained('ethz')]
    models_arg = [{"name": "OBSTransformer", "dataset": 'obst2024'}, {"name": "GPD", "dataset": 'ethz'}]

    for dataset in datasets:
        models.append(sbm.EQTransformer.from_pretrained(dataset))
        models_arg.append({"name": "EQTransformer", "dataset": dataset})
        models.append(sbm.PhaseNet.from_pretrained(dataset))
        models_arg.append({"name": "PhaseNet", "dataset": dataset})
        models.append(sbm.GPD.from_pretrained(dataset))
        models_arg.append({"name": "GPD", "dataset": dataset})

    for model in models:
        model.cuda()

    # pred
    picks = []
    for i, model in enumerate(models):
        if models_arg[i]["name"] == "GPD":
            model_preds = model.annotate(stream)
        else:
            model_preds = model.annotate(stream, blinding=(0, 0))
        # stream.plot()
        # model_preds.plot()
        picks.append(find_ps_peaks(model_preds, stream[0].stats.starttime))


    P_pick = []
    S_pick = []
    for pick in picks:
        P_pick.append(pick["P"])
        S_pick.append(pick["S"])

    p_flag, p_dict = average_arrival(P_pick, "P",p_win_s=0.2)
    s_flag, s_dict = average_arrival(S_pick, "S",s_win_s=0.3)

    if p_flag and s_flag:
        p = round(p_dict["arrival"] * 100, 1)
        s = round(s_dict["arrival"] * 100, 1)
        return True, p, s
    elif p_flag:
        p = round(p_dict["arrival"] * 100, 1)
        return True, p, None
    else:
        return False, None, None


