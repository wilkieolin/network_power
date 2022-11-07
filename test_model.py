import subprocess as sp
import tensorflow as tf
import argparse
import pickle as p
import pandas as pd
from timeit import timeit

JULIA_PATH = "/home/wolinammentorp/julia-1.8.2/bin/julia"
DEVICE_NAME = "NVIDIA A100 40 GB"
IDLE_POWER_FILENAME = "idle_power.csv"
ACTIVE_POWER_FILENAME = "power_output.csv"

parser = argparse.ArgumentParser(description="Options")
#parser.add_argument('--output', type = str, default = "run.csv")
parser.add_argument('--model', type = str, default = "EfficientNetB0")
parser.add_argument('--samples', type = int, default = 100)
args = parser.parse_args()


def launch_power_check(sample_time: float = 2.00, 
                    poll_time = 0.010, 
                    filename: str = "power_output.csv"):

    return sp.Popen([JULIA_PATH, 
                "track_gpu_power.jl", 
                "--sample_time=" + sample_time,
                "--poll_time=" + poll_time,
                "--filename=" + filename,
                ])
    
def setup(model: str = "EfficientNetB0", image_shape: tuple = (400, 400, 1)):
    n_batch = 1
    shape = (n_batch, *image_shape)
    sample = tf.random.uniform(shape)

    #get the standard keras model specified by the args
    model_fn = getattr(tf.keras.applications, model)
    model = model_fn(weights = None, include_top = False, input_shape = image_shape)
    call_fn = lambda: model(sample)

    #call it once to build the model and transfer to gpu
    call_fn()
    print("Setup done")
    return call_fn


def test_latency(call_fn, n_samples: int):
    t_total = timeit(call_fn, number = n_samples)
    t_avg = t_total / n_samples

    print("Total time elapsed", t_total)
    return t_avg

#check the GPU idle power for baseline
proc = launch_power_check(filename=IDLE_POWER_FILENAME)
proc.wait()
#initialize the neural network on device
call = setup(model = args.model)
#launch the power tracking program asynchronously
proc = launch_power_check(filename=ACTIVE_POWER_FILENAME)
#repeatedly call the inference function to test latency
t_avg = test_latency(call, args.samples)
print("Average latency", t_avg)
#kill the power tracking program if it's still running
proc.terminate()

#load and format the data to a single file
filename = "test_" + args.model + ".p"
idle_power = pd.read_csv(IDLE_POWER_FILENAME)
power = pd.read_csv(ACTIVE_POWER_FILENAME)

data = {"latency" : t_avg,
        "samples" : args.samples,
        "device" : DEVICE_NAME,
        "model" : args.model,
        "active power" : power,
        "idle power" : idle_power,}
p.dump(data, open(filename, "wb"))