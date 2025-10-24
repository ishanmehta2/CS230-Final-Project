# use this as a sample file on how to load in the data and see the columns

from datasets import load_dataset

ds = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned")

print(ds['train'][0]['source'])
count = 0
for i in range(100):
    if ds['train'][0]['source'] == 'evol_instruct':
        count += 1
print(count)