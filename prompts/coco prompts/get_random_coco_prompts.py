import pandas as pd


def get_random_prompts(file_address, N=1000):
    df = pd.read_csv(file_address)
    random_df = df.sample(n=N, random_state=42)
    random_df.to_csv(f"random_{file_address}", index=False)
    return


get_random_prompts("coco_train_two_objects.csv")
get_random_prompts("coco_train_three_objects.csv")