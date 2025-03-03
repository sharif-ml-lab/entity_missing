import pandas as pd
import numpy as np


def generate_two_objects_prompts(entities, file_name):
    prompts = []
    for i in range(len(entities)):
        for j in range(i+1, len(entities)):
            t_1 = 'an' if entities[i][0] in ['a', 'u', 'i', 'o', 'e'] else 'a'
            t_2 = 'an' if entities[j][0] in ['a', 'u', 'i', 'o', 'e'] else 'a' 
            prompts.append(f"{t_1} {entities[i]} and {t_2} {entities[j]}")

    pd.DataFrame(data={"prompt": prompts}).to_csv(f"{file_name}", index=False)



def generate_three_objects_prompts(entities):
    prompts = []
    for i in range(len(entities)):
        for j in range(i+1, len(entities)):
            for k in range(j+1, len(entities)):
                t_1 = 'an' if entities[i][0] in ['a', 'u', 'i', 'o', 'e'] else 'a'
                t_2 = 'an' if entities[j][0] in ['a', 'u', 'i', 'o', 'e'] else 'a'
                t_3 = 'an' if entities[k][0] in ['a', 'u', 'i', 'o', 'e'] else 'a'
                prompts.append(f"{t_1} {entities[i]} and {t_2} {entities[j]} and {t_3} {entities[k]}")
    
    random_prompts = np.random.choice(prompts, size=1000, replace=False)
    
    pd.DataFrame(data={"prompt": random_prompts}).to_csv("random_three_objects_prompts.csv", index=False)
    pd.DataFrame(data={"prompt": prompts}).to_csv("three_objects_prompts.csv", index=False) 



def generate_four_objects_prompts(entities, n_prompts):
    prompts = []
    for i in range(len(entities)):
        for j in range(i+1, len(entities)):
            for k in range(j+1, len(entities)):
                for l in range(k+1, len(entities)):
                    t_1 = 'an' if entities[i][0] in ['a', 'u', 'i', 'o', 'e'] else 'a'
                    t_2 = 'an' if entities[j][0] in ['a', 'u', 'i', 'o', 'e'] else 'a'
                    t_3 = 'an' if entities[k][0] in ['a', 'u', 'i', 'o', 'e'] else 'a'
                    t_4 = 'an' if entities[l][0] in ['a', 'u', 'i', 'o', 'e'] else 'a'
                    prompts.append(f"{t_1} {entities[i]} and {t_2} {entities[j]} and {t_3} {entities[k]} and {t_4} {entities[l]}")
    
    random_prompts = np.random.choice(prompts, size=n_prompts, replace=False)
    
    pd.DataFrame(data={"prompt": random_prompts}).to_csv("random_four_objects_prompts.csv", index=False)
    pd.DataFrame(data={"prompt": prompts}).to_csv("four_objects_prompts.csv", index=False) 



entities = sorted(pd.read_csv("../entities/entities.csv")["entity"].tolist())
validation_entities = sorted(pd.read_csv("../entities/validation_entities.csv")["entity"].tolist())
test_entities = sorted(pd.read_csv("../entities/test_entities.csv")["entity"].tolist())
# generate_three_objects_prompts(test_entities)
generate_four_objects_prompts(test_entities, 500)
# generate_two_objects_prompts(validation_entities, "validation_two_objects_prompts.csv")
# generate_two_objects_prompts(test_entities, "test_two_objects_prompts.csv")

