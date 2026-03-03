from src.input.input import get_partition

df = get_partition(partition="small-g", type="with_features")

print(df.shape[0])
