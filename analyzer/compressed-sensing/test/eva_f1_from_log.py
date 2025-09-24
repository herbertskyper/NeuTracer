import re

def parse_metrics(lines):
    precisions, recalls, f1s = [], [], []
    for line in lines:
        m = re.search(r'Precision=([\d.]+), Recall=([\d.]+), F1=([\d.]+)', line)
        if m:
            precisions.append(float(m.group(1)))
            recalls.append(float(m.group(2)))
            f1s.append(float(m.group(3)))
    return precisions, recalls, f1s

with open('test_result.txt', encoding='utf-8') as f:
    lines = f.readlines()

dataset_metrics = {}
current_dataset = None
for line in lines:
    line = line.strip()
    if line.startswith('Dataset'):
        current_dataset = line
        dataset_metrics[current_dataset] = []
    elif line.startswith('service'):
        dataset_metrics[current_dataset].append(line)

for ds, records in dataset_metrics.items():
    precisions, recalls, f1s = parse_metrics(records)
    if precisions:
        print(f"{ds}:")
        print(f"  Avg Precision: {sum(precisions)/len(precisions):.4f}")
        print(f"  Avg Recall:    {sum(recalls)/len(recalls):.4f}")
        print(f"  Avg F1:        {sum(f1s)/len(f1s):.4f}")