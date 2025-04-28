import subprocess
import time
import re


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


def get_cpu_temp():
    try:
        result = subprocess.run(["sensors"], capture_output=True, text=True)
        output = result.stdout

        match = re.search(r"Tctl:\s+\+([0-9]+)\.?\d*°C", output)
        if match:
            temp = int(match.group(1))
            return temp
        else:
            print("[CPU] Failed to parse CPU temp from sensors output.")
            return -1
    except Exception as e:
        print(f"[CPU] Failed to get CPU temp: {e}")
        return -1


def wait_for_cooldown(gpu_id=0, threshold=77, gpu_threshold=65, cpu_threshold=65):
    gpu_temp = get_gpu_temp(gpu_id)
    cpu_temp = get_cpu_temp()

    if gpu_temp >= threshold or cpu_temp >= threshold:

        print(f"[System] Waiting for Cooldown | GPU: {gpu_temp} | CPU: {cpu_temp}", flush=True)
        while True:
            gpu_temp = get_gpu_temp(gpu_id)
            cpu_temp = get_cpu_temp()

            print(f"Current GPU temp: {gpu_temp}℃")
            print(f"Current CPU temp: {cpu_temp}℃")

            if gpu_temp <= gpu_threshold and cpu_temp <= cpu_threshold:
                print("[System] Temps are low enough. Resuming operation.", flush=True)
                break

            time.sleep(10)


if __name__ == '__main__':
    print(get_gpu_temp(0))
    print(get_gpu_temp(1))
    print(get_cpu_temp())
