import os
import random
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchcubicspline import(natural_cubic_spline_coeffs, NaturalCubicSpline)

from model_function import *
from utils import *

from pysteps import motion


def flow_field(
    frames, 
    method="proesmans", 
    wrap=8, 
    gain=255.0, 
    **kw
):
    x = np.ascontiguousarray(frames, dtype=np.float64) * gain
    x = np.concatenate([x[..., -wrap:], x, x[..., :wrap]], axis=-1)
    uv = motion.get_method(method)(x, **kw)
    return uv[..., wrap:-wrap] 


def fit_velocity_pysteps(
    time_idx, 
    data_loader, 
    Final_train_data, 
    num_years,
    paths_to_data, 
    scale, 
    H, 
    W, 
    types,
    method="proesmans", 
    lag=1, 
    alpha=None, 
    **flow_kw,
):
    C = len(paths_to_data) * (scale + 1)
    out = []
    for idx_steps, _ in zip(time_idx, data_loader):
        t0 = idx_steps[0].item()
        idxs = [t0 - i for i in range(lag, -1, -1)]        # lag=1 -> [t0-1, t0]

        # explizit stapeln + permutieren, NICHT view() (siehe Abschnitt 5)
        f = torch.stack([Final_train_data[j].view(num_years, C, H, W) for j in idxs])
        f = f.permute(1, 2, 0, 3, 4).cpu().numpy()          # (years, C, T, H, W)

        v = np.empty((num_years, 2, C, H, W), dtype=np.float32)
        for y in range(num_years):
            for c in range(C):
                uv = flow_field(f[y, c], method=method, **flow_kw)
                v[y, 0, c], v[y, 1, c] = uv[0], uv[1]
        out.append(v)

    Final_v = (-1) * (np.stack(out) / 6.0)   # (B, years, 2, C, H, W)
    if alpha is not None:
        Final_v = Final_v * np.asarray(alpha).reshape(1, 1, 1, C, 1, 1)
    np.save(f'initial_velocities/optical_flow_vel_vet/{types}_{method}_vel.npy', Final_v)
    return Final_v


if __name__ == '__main__':
    device = 'cuda:0'
    set_seed(42)

    cwd = os.getcwd()
    data_dir = ''

    train_time_scale= slice('2006','2016')
    val_time_scale = slice('2016','2016')
    test_time_scale = slice('2017','2018')

    paths_to_data = [
        str(data_dir) + '/era5_data/geopotential_500/*.nc',
        str(data_dir) + '/era5_data/temperature_850/*.nc',
        str(data_dir) + '/era5_data/2m_temperature/*.nc',
        str(data_dir) + '/era5_data/10m_u_component_of_wind/*.nc',
        str(data_dir) + '/era5_data/10m_v_component_of_wind/*.nc'
    ]
    const_info_path = [str(data_dir) +  '/era5_data/constants/constants_5.625deg.nc']

    # This variable is unclear
    levels = ["z","t","t2m","u10","v10"]

    assert (
        len(paths_to_data) == len(levels)
    ), "Paths to different type of data must be same as number of types of observations"

    Final_train_data = 0
    Final_val_data = 0
    Final_test_data = 0
    max_lev = []
    min_lev = []

    for idx,data in enumerate(paths_to_data):
        Train_data, Val_data, Test_data, time_steps, lat, lon, mean, std, time_stamp = get_train_test_data_without_scales_batched(
            data,
            train_time_scale,
            val_time_scale,
            test_time_scale,
            levels[idx],
            # args.spectral
            0,
        )  
        max_lev.append(mean)
        min_lev.append(std)
        if idx==0: 
            Final_train_data = Train_data
            Final_val_data = Val_data
            Final_test_data = Test_data
        else:
            Final_train_data = torch.cat([Final_train_data,Train_data],dim=2)
            Final_val_data = torch.cat([Final_val_data,Val_data],dim=2)
            Final_test_data = torch.cat([Final_test_data,Test_data],dim=2)

    H,W = Train_data.shape[3],Train_data.shape[4]
    batch_size = 8
    Train_loader = DataLoader(Final_train_data[2:],batch_size=batch_size,shuffle=False,pin_memory=False)
    Val_loader = DataLoader(Final_val_data[2:],batch_size=batch_size,shuffle=False,pin_memory=False)
    Test_loader = DataLoader(Final_test_data[2:],batch_size=batch_size,shuffle=False,pin_memory=False)
    time_loader = DataLoader(time_steps[2:],batch_size=batch_size,shuffle=False,pin_memory=False)
    time_idx_steps = torch.tensor([i for i in range(365*4)]).view(-1,1)
    time_idx = DataLoader(time_idx_steps[2:],batch_size=batch_size,shuffle=False,pin_memory=False)
    scale = 0

    fit_velocity_pysteps(
        time_idx=time_idx,
        data_loader=Train_loader,
        Final_train_data=Final_train_data,
        num_years=10,
        paths_to_data=paths_to_data,
        scale=scale,
        H=H,
        W=W,
        types='train_10year_2day_mm',
        method='VET',
    )
    fit_velocity_pysteps(
        time_idx=time_idx,
        data_loader=Val_loader,
        Final_train_data=Final_val_data,
        num_years=1,
        paths_to_data=paths_to_data,
        scale=scale,
        H=H,
        W=W,
        types='val_10year_2day_mm',
        method='VET',
    )
    fit_velocity_pysteps(
        time_idx=time_idx,
        data_loader=Test_loader,
        Final_train_data=Final_test_data,
        num_years=2,
        paths_to_data=paths_to_data,
        scale=scale,
        H=H,
        W=W,
        types='test_10year_2day_mm',
        method='VET',
    )
