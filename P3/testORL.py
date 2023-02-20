CUDA_LAUNCH_BLOCKING=1

from Settings import Settings
import matplotlib.pyplot as plt

t = Settings(sRandSeed=42, sDataset="ORL")
res = t.load_run_save_evaluate()
print(f'Accurarcy is: {res * 100}%')
