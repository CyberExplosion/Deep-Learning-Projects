from Settings import Settings

t = Settings(sRandSeed=31, sDataset="cora", sUseSave=False)
res = t.load_run_save_evaluate()
print(f'Accurarcy is: {res * 100}%')