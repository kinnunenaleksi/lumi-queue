from train.train import train_models, combine_results
import datetime

train_models()

model = "rf-gb.largemem-lumid.20260304T1910"

comb_res, comb_dfs = combine_results(f"results/{model}")

print(comb_dfs.get("accuracy_metrics"))
