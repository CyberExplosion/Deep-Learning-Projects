from Settings import Settings

t = Settings(sRandSeed=42, sDataset="citeseer", sUseSave=False)
res = t.load_run_save_evaluate()
print(f'Accurarcy is: {res * 100}%')