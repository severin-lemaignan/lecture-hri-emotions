import csv
import glob
import os.path

TRAINING_FACES_PER_EMOTION = 50

emotions=["fear","happiness","anger","surprise","sadness","disgust"]

au_keys = ["AU01","AU02","AU04","AU05","AU06","AU07","AU09","AU10","AU12","AU14","AU15","AU17","AU20","AU23","AU25","AU26","AU45"]

dataset = {}

global_idx = 0

for emotion in emotions:
    faces = glob.glob("%s/processed/*.csv" % emotion)

    for face in faces:
        with open(face, 'rb') as csvfile:
            details = csv.DictReader(csvfile, skipinitialspace=True)
            for row in details:
                dataset[global_idx] = {"emotion":emotions.index(emotion),
                        "face": os.path.dirname(face) + "/aligned/" + os.path.basename(face)[:-4] + "_aligned.bmp"}
                for au in au_keys:
                    dataset[global_idx][au] = row[au+"_r"] if float(row[au+"_c"]) == 1 else 0

            global_idx += 1

with open("emotions_action_units_full.csv", "w") as output_csv:
    writer = csv.DictWriter(output_csv, fieldnames=["emotion"] + au_keys + ["face"])

    writer.writeheader()
    for k, v in dataset.items():
        writer.writerow(v)

with open("emotions_action_units_training.csv", "w") as output_csv:
    writer = csv.DictWriter(output_csv, fieldnames=["emotion"] + au_keys + ["face"])

    writer.writeheader()

    idxs = {}
    for k, v in dataset.items():
        idxs[v["emotion"]] = idxs.setdefault(v["emotion"],0) + 1
        if idxs[v["emotion"]] <= TRAINING_FACES_PER_EMOTION:
            writer.writerow(v)

with open("emotions_action_units_test.csv", "w") as output_csv:
    writer = csv.DictWriter(output_csv, fieldnames=["emotion"] + au_keys + ["face"])

    writer.writeheader()

    idxs = {}
    for k, v in dataset.items():
        idxs[v["emotion"]] = idxs.setdefault(v["emotion"],0) + 1
        if idxs[v["emotion"]] > TRAINING_FACES_PER_EMOTION:
            writer.writerow(v)

