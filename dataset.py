import pandas as pd
from datasets import load_dataset

dataset = load_dataset('medical_dialog', 'processed.en')
df = pd.DataFrame(dataset['train'])

dialog = []

# 患者と医者の発言をそれぞれ抽出した後、順にリストに格納
patient, doctor = zip(*df['utterances'])
for i in range(len(patient)):
  dialog.append(patient[i])
  dialog.append(doctor[i])

df_dialog = pd.DataFrame({"dialog": dialog})

# 成形終了したデータセットを保存
df_dialog.to_csv('medical_data.txt', sep=' ', index=False)
