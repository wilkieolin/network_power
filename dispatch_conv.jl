
models = ["EfficientNetB" * string(i) for i in 0:7]
device = 1

commands = [`python test_model.py --cuda_device=$device --model=$model` for model in models]
map(run, commands)
