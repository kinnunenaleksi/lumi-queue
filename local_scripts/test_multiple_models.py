from train.train import train_models, combine_results

train_models()

comb_res, comb_dfs = combine_results()

print(comb_dfs.get("accuracy_metrics"))
# print(res.get("res_largemem_rf_baseline"))
