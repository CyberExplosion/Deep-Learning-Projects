from Setting_MLP import Setting_MLP
import matplotlib.pyplot as plt

t = Setting_MLP()
print(t.method.layers)
res = t.load_run_save_evaluate()