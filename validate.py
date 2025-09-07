import re
import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

# === Parse log file ===
with open("/home/rodolfo/Desktop/training-resnet50.txt", "r") as f:
    lines = f.readlines()

train_pattern = re.compile(
    r"Epoch (\d+) \| Train Loss: ([\d.]+) \| Train IoU: ([\d.]+) \| Train Acc: ([\d.]+) "
    r"\| Train Precision: ([\d.]+) \| Train Recall: ([\d.]+) \| Train F1-score: ([\d.]+)"
)

val_pattern = re.compile(
    r"Epoch (\d+) \| Val IoU: ([\d.]+) \| Val Acc: ([\d.]+) "
    r"\| Val Precision: ([\d.]+) \| Val Recall: ([\d.]+) \| Val F1-score: ([\d.]+)"
)

records = {}
for line in lines:
    t_match = train_pattern.search(line)
    v_match = val_pattern.search(line)

    if t_match:
        epoch = int(t_match.group(1))
        records.setdefault(epoch, {})
        records[epoch].update({
            "Train Loss": float(t_match.group(2)),
            "Train IoU": float(t_match.group(3)),
            "Train Acc": float(t_match.group(4)),
            "Train Prec": float(t_match.group(5)),
            "Train Rec": float(t_match.group(6)),
            "Train F1": float(t_match.group(7)),
        })

    if v_match:
        epoch = int(v_match.group(1))
        records.setdefault(epoch, {})
        records[epoch].update({
            "Val IoU": float(v_match.group(2)),
            "Val Acc": float(v_match.group(3)),
            "Val Prec": float(v_match.group(4)),
            "Val Rec": float(v_match.group(5)),
            "Val F1": float(v_match.group(6)),
        })

# === Convert to DataFrame ===
df = pd.DataFrame.from_dict(records, orient="index").sort_index()
df.index.name = "Epoch"
df.to_csv("training_metrics.csv")
logging.info(">>>> Metrics saved to training_metrics.csv")

# === Plotting ===
epochs = df.index.tolist()
plt.figure(figsize=(18, 10))

# IoU
plt.subplot(2, 3, 1)
plt.plot(epochs, df["Train IoU"], "b-o", label="Train IoU")
plt.plot(epochs, df["Val IoU"], "r--o", label="Val IoU")
plt.title("IoU over Epochs")
plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.legend()

# Accuracy
plt.subplot(2, 3, 2)
plt.plot(epochs, df["Train Acc"], "b-o", label="Train Acc")
plt.plot(epochs, df["Val Acc"], "r--o", label="Val Acc")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Precision
plt.subplot(2, 3, 3)
plt.plot(epochs, df["Train Prec"], "b-o", label="Train Prec")
plt.plot(epochs, df["Val Prec"], "r--o", label="Val Prec")
plt.title("Precision over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.legend()

# Recall
plt.subplot(2, 3, 4)
plt.plot(epochs, df["Train Rec"], "b-o", label="Train Recall")
plt.plot(epochs, df["Val Rec"], "r--o", label="Val Recall")
plt.title("Recall over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.legend()

# F1-score
plt.subplot(2, 3, 5)
plt.plot(epochs, df["Train F1"], "b-o", label="Train F1")
plt.plot(epochs, df["Val F1"], "r--o", label="Val F1")
plt.title("F1-score over Epochs")
plt.xlabel("Epoch")
plt.ylabel("F1-score")
plt.legend()

save_path = "training_metrics.svg"

plt.tight_layout()
plt.savefig(save_path, dpi=150, format="svg")
logging.info(f">>>> Training plot saved to {save_path}")
plt.close()
