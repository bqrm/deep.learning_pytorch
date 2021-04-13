import pynvml

pynvml.nvmlInit()
num_gpu = pynvml.nvmlDeviceGetCount()

print()
print("{} gpu(s) in total\n".format(num_gpu))
print("memory on gpu:")
print(" id |     total    |     free     |      used    |       name")

for gpu_id in range(num_gpu):
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)   # gpu_id
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

    print(" {:2d} | {:12d} | {:12d} | {:12d} | {}".format(gpu_id, meminfo.total, meminfo.free, meminfo.used, pynvml.nvmlDeviceGetName(handle).decode()))










