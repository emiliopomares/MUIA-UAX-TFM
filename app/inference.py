import torch
import threading
import time
from tensor_permutation import unpermute_target_tensor, crop_output

import numpy as np

def occ_to_lists(occ):
    x = []
    y = []
    z = []
    Ni, Nj, Nk = occ.shape
    for i in range(Ni):
        for j in range(Nj):
            for k in range(Nk):
                if occ[i, j, k] > 0:
                    x.append(j)
                    y.append(i)
                    z.append(k)
    return np.array(x), np.array(y), np.array(z)

def run_inference(model, input_data, occ_threshold=0.75):
    print("              >>>> Running inference sync ")
    model.eval()
    with torch.no_grad():
        # Perform inference
        start_time = time.time()
        pred = model(input_data)
        end_time = time.time()
        print(f"\n\n      Inference time: {int((end_time-start_time)*1000)}ms \n\n")
        occ = crop_output(unpermute_target_tensor(torch.sigmoid(pred)[0]))
        occ = (occ-torch.min(occ)) / (torch.max(occ)-torch.min(occ))
        occ = (occ > occ_threshold) * 1.0
        return occ

def run_inference_cb(model, input_data, callback, occ_threshold=0.25):
    print("              >>>> Running inference with callback")
    model.eval()
    with torch.no_grad():
        # Perform inference
        start_time = time.time()
        pred = model(input_data)
        end_time = time.time()
        print(f"\n\n      Inference time: {int((end_time-start_time)*1000)}ms \n\n")
        occ = crop_output(unpermute_target_tensor(torch.sigmoid(pred)[0]))
        occ = (occ-torch.min(occ)) / (torch.max(occ)-torch.min(occ))
        occ = (occ > occ_threshold) * occ
        print("              >>>> Inference done ")
        callback(*occ_to_lists(occ))

def run_points(model, input_data, callback, occ_threshold=0.25):
    print(f" shape {input_data.shape}")
    i = 0
    while True:
       run_inference_cb(model, input_data[i:i+1], callback, occ_threshold) 
       i = (i+1) % input_data.shape[0]

def run_inference_thread(model, input_data, callback, occ_threshold=0.25):
    # Create and start the inference thread
    print("              >>>> Starting thread ")
    thread = threading.Thread(target=run_points, args=(model, input_data, callback, occ_threshold))
    thread.start()