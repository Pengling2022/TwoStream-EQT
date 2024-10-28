import os

import h5py
import numpy as np
from obspy import Stream, Trace
from scipy.signal import find_peaks
from scipy.stats import zscore
import seisbench.models as sbm
def crop_dataset(data, p_index, s_index, length=6000):
    min_pre = min(p_index, 100)

    n = int((s_index - p_index) * 1.4)   # 事件长度
    if n >= length:
        q = np.random.uniform(1.05, length/(s_index - p_index)  -0.05)
        n = int((s_index - p_index) * q)

    m = length - n  # 剩余长度
    # np.random.seed(1)
    # Determine start index
    if min(m, p_index) <= min_pre:
        pre_index = min(m, p_index)
    else:
        pre_index = np.random.randint(min_pre, min(m, p_index))
    start_index = int(p_index - pre_index)

    # Determine end index
    end_index = int(p_index + n + m -pre_index)

    if end_index - start_index == length:
        # Crop dataset

        cropped_data = data[start_index:end_index, :]

        # Calculate new P and S indices
        new_p_index = p_index - start_index
        new_s_index = s_index - start_index

        # Check if cropped data length is less than specified length, pad with zeros if necessary
        if cropped_data.shape[0] < length:
            pad_width = ((0, length - cropped_data.shape[0]), (0, 0))
            cropped_data = np.pad(cropped_data, pad_width, mode='constant', constant_values=0)

        # Filter
        channels = ["HHZ", "HHN", "HHE"]
        stream = Stream([Trace(data=col, header={"sampling_rate": 50, "channel": channels[i]}) for i, col in
                         enumerate(cropped_data.T)])

        stream.detrend()
        stream.resample(100)
        stream.filter('bandpass', freqmin=1, freqmax=45, corners=4, zerophase=True)
        stream.normalize()

        return stream, float(new_p_index)*2, float(new_s_index)*2, pre_index

    else:
        raise ValueError("S index not included in cropped data!")

def get_from_DiTing(part=None,
                    key=None,
                    h5file_path=None, ):
    # with h5py.File(h5file_path + '/DiTing330km_part_{}.hdf5'.format(part), 'r') as f:
    with h5py.File(h5file_path + '/waveforms_{}.hdf5'.format(part), 'r') as f:
        #     dataset = f.get('earthquake/' + str(key))
        #     data = np.array(dataset).astype(np.float32).T

        dataset = f.get('earthquake/' + str(key))
        data = np.array(dataset).astype(np.float32)

        return data

def get_from_DiTing_noise(part=None, key=None, h5file_folder=None):
    """
    Input:
    part, key, h5file_folder

    Output:
    filtered_data
    """
    # h5file_path = os.path.join(h5file_folder, 'DiTing330km_part_{}.hdf5'.format(part))
    h5file_path = os.path.join(h5file_folder, 'noise_{}.hdf5'.format(part))
    with h5py.File(h5file_path, 'r') as f:
        dataset = f.get('noise/' + str(key))
        data = np.array(dataset).astype(np.float32)
    return data

def make_labels(data, p_t, s_t):
    nt = data.shape[0]
    event = np.zeros(shape=(nt, 1))

    # 信号标签
    event_start = int(p_t)
    event_end = int(p_t + int((s_t - p_t) * 1.4) + 1)
    event[event_start:event_end] = 1

    # 到时标签
    time_steps = np.arange(nt)
    gaussian_std_dev = 20
    P_arrival = np.exp(-((time_steps - p_t) ** 2) / (2 * gaussian_std_dev ** 2))
    S_arrival = np.exp(-((time_steps - s_t) ** 2) / (2 * gaussian_std_dev ** 2))

    return event, P_arrival, S_arrival

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

def average_arrival_noise(picks,type,p_win_s=0.3,s_win_s=0.5,z_threshold=2):
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
    if len(picks)>=np.floor(((n+1)/3)) and np.max(picks)-np.min(picks)<time_dict[type]:
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

def predict_DiTing(stream):
    datasets = ["scedc", "stead"]  # ,'ethz',,'geofon'
    models = [sbm.OBSTransformer.from_pretrained("obst2024"), sbm.GPD.from_pretrained('ethz')]
    models_arg = [{"name": "OBSTransformer", "dataset": 'obst2024'}, {"name": "GPD", "dataset": 'ethz'}]

    for dataset in datasets:
        models.append(sbm.EQTransformer.from_pretrained(dataset))
        models_arg.append({"name": "EQTransformer", "dataset": dataset})
        models.append(sbm.PhaseNet.from_pretrained(dataset))
        models_arg.append({"name": "PhaseNet", "dataset": dataset})
        # models.append(sbm.GPD.from_pretrained(dataset))
        # models_arg.append({"name": "GPD", "dataset": dataset})

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

def predict_DiTing_noise(stream):
    datasets = ["scedc", "stead"]  # ,'ethz',,'geofon'
    models = [sbm.OBSTransformer.from_pretrained("obst2024"), sbm.GPD.from_pretrained('ethz')]
    models_arg = [{"name": "OBSTransformer", "dataset": 'obst2024'}, {"name": "GPD", "dataset": 'ethz'}]

    for dataset in datasets:
        models.append(sbm.EQTransformer.from_pretrained(dataset))
        models_arg.append({"name": "EQTransformer", "dataset": dataset})
        models.append(sbm.PhaseNet.from_pretrained(dataset))
        models_arg.append({"name": "PhaseNet", "dataset": dataset})
        # models.append(sbm.GPD.from_pretrained(dataset))
        # models_arg.append({"name": "GPD", "dataset": dataset})

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
        picks.append(find_ps_peaks(model_preds, stream[0].stats.starttime,p_th=0.1,s_th=0.1))


    P_pick = []
    S_pick = []
    for pick in picks:
        P_pick.append(pick["P"])
        S_pick.append(pick["S"])

    p_flag, p_dict = average_arrival_noise(P_pick, "P",p_win_s=0.2)
    s_flag, s_dict = average_arrival_noise(S_pick, "S",s_win_s=0.3)

    if p_flag and s_flag:
        p = round(p_dict["arrival"] * 100, 1)
        s = round(s_dict["arrival"] * 100, 1)
        return True, p, s
    elif p_flag:
        p = round(p_dict["arrival"] * 100, 1)
        return True, p, None
    else:
        return False, None, None