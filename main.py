import os

gpu_id = 0
# gpu_id = 1
# gpu_id = 2
# gpu_id = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

if __name__ == "__main__":
    from trainConvGAN import trainConvGanMain
    trainConvGanMain()