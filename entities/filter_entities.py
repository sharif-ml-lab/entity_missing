import pandas as pd
import numpy as np


coco_entities = pd.read_csv("./raw_entities/COCO.csv")["COCO"].tolist()

raw_entities = coco_entities

filtered_entities = list(set(raw_entities))

filtered_entities = [entity for entity in filtered_entities if len(entity.split())==1]

plural_entities = [entity for entity in filtered_entities if entity[-1] == "s" and entity not in ["bus"]]

filtered_entities = [entity for entity in filtered_entities if entity not in plural_entities]

filtered_entities = list(set(filtered_entities))

np.random.seed(42)
validation_entities = list(np.random.choice(filtered_entities, 20, replace=False))

test_entities = list(set(filtered_entities) - set(validation_entities))

pd.DataFrame(data={"entity": filtered_entities}).to_csv("./entities.csv", index=False)
pd.DataFrame(data={"entity": validation_entities}).to_csv("./validation_entities.csv", index=False)
pd.DataFrame(data={"entity": test_entities}).to_csv("./test_entities.csv", index=False)