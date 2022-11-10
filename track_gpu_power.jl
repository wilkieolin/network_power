using ArgParse

arg_table = ArgParseSettings()

@add_arg_table arg_table begin
    "--sample_time"
        help = "Amount of time to run power sampling"
        default = 2.00
        arg_type = Float64
    "--poll_time"
        help = "Amount of time to delay between sampling power"
        default = 0.010
        arg_type = Float64
    "--filename"
        help = "Name of the file to write output to"
        default = "power_output.csv"
    "--cuda_device"
        help = "CUDA device to track power on"
        default = 0
        arg_type = Int64
end

#load arguments parsed from the command line
args = parse_args(ARGS, arg_table)
poll_time = args["poll_time"]
max_time = args["sample_time"]
cuda_device = args["cuda_device"]
max_samples = Int(floor(max_time / poll_time))
filename = args["filename"]


#define commands
nvs = `nvidia-smi --query-gpu=power.draw --format=csv`

#define the power regex
power_regex = r"(\d+.\d+) W"


function write_to_file()
    open(filename, "w") do io
        for p in p_samples
            write(io, p * ",")
        end
    end
end

p_samples = []
#run up to a certain time limit
for i in 1:max_samples
    s = read(nvs, String)
    m = match(power_regex, s)

    #return -1 for capture error
    if m === nothing
        append!(p_samples, -1)
    else
        #offset device by 1 for index-by-one
        capture = String(m.captures[cuda_device+1])
        append!(p_samples, [capture])
    end

    write_to_file()
    sleep(poll_time)
end

#print(p_samples)