from Settings import Settings
import matplotlib.pyplot as plt

t = Settings(sRandSeed=42, sDataset="CIFAR")
res = t.load_run_save_evaluate()
print(f'Accurarcy is: {res * 100}%')
