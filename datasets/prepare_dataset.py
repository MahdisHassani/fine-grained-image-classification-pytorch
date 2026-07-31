from scipy.io import loadmat
import json
import pandas as pd

train_annos = loadmat("devkit/cars_train_annos.mat")
print(train_annos.keys())

annotations = train_annos["annotations"]
print(type(annotations))
print(annotations.shape)
print(len(annotations[0]))


meta = loadmat("devkit/cars_meta.mat")
class_names = meta["class_names"][0]

print(len(class_names))
print(class_names[0][0])

rows = []

for anno in annotations[0]:

    x1 = int(anno["bbox_x1"][0, 0])

    y1 = int(anno["bbox_y1"][0, 0])

    x2 = int(anno["bbox_x2"][0, 0])

    y2 = int(anno["bbox_y2"][0, 0])

    class_id = int(anno["class"][0, 0])

    filename = anno["fname"][0]

    class_name = class_names[class_id - 1][0]

    rows.append([
        filename,
        class_id,
        class_name,
        x1,
        y1,
        x2,
        y2
    ])

df = pd.DataFrame(
    rows,
    columns=[
        "filename",
        "class_id",
        "class_name",
        "x1",
        "y1",
        "x2",
        "y2"
    ]
)

print(df.head())

df.to_csv("outputs/train.csv", index=False)


meta = loadmat("devkit/cars_meta.mat")

class_names = [c[0] for c in meta["class_names"][0]]

with open("outputs/class_names.json", "w") as f:
    json.dump(class_names, f, indent=4)