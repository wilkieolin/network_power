
models = ["EfficientNetB" * string(i) for i in 7:7]

commands = [`python test_model.py --model=$model` for model in models]
map(run, commands)