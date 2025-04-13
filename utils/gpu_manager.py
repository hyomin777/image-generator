
import subprocess
import time


def get_gpu_temp(gpu_id=0):
    try:
        result = subprocess.run([
            "nvidia-smi",
            "--query-gpu=temperature.gpu",
            f"--id={gpu_id}",
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True)
        temp = int(result.stdout.strip().split("\n")[0])
        return temp
    except Exception as e:
        print(f"[GPU {gpu_id}] Failed to get GPU temp: {e}")
        return -1


def wait_for_cooldown(gpu_id=0, threshold=70):
    print(f"[GPU {gpu_id}]aiting to cool below {threshold}C..", flush=True)
    while True:
        temp = get_gpu_temperature(gpu_id)
        print(f"   → Current temp: {temp}℃")
        if temp <= threshold:
            print(f"[GPU {gpu_id}] Resuming training.", flush=True)
            break
        time.sleep(10)
