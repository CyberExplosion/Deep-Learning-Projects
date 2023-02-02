from Setting_MLP import Setting_MLP

t = Setting_MLP(extraSettingsPath='P2/hiddenLayers.json')
print(t.method.layers)
t.load_run_save_evaluate()
