from Setting_MLP import Setting_MLP
import matplotlib.pyplot as plt

t = Setting_MLP(sRandSeed=42)
print(t.method.layers)
res = t.load_run_save_evaluate()
print(f'Accurarcy is: {res * 100}%')
