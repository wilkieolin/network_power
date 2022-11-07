using ArgParse

@add_arg_table arg_table begin
    "--sample_time"
        help = "Amount of time to run power sampling"
        default = 2.00
    "--poll_time"
        help = "Amount of time to delay between sampling power"
        default = 0.010
    "--filename"
        help = "Name of the file to write output to"
        default = "power_output.csv"
end

#load arguments parsed from the command line
args = parse_args(ARGS, arg_table)
poll_time = args["poll_time"]
max_time = args["sample_time"]
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
    if m == nothing
        append!(p_samples, -1)
    else
        capture = String(m.captures[1])
        append!(p_samples, [capture])
    end

    write_to_file()
    sleep(poll_time)
end

#print(p_samples)