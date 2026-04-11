from typing import Any
import torch
from pandas import DataFrame
import numpy as np
from sklearn.metrics import accuracy_score


def PFI_accuracies(
    model: Any, latent_dataframe: DataFrame, num_concepts: int, repeat: int = 100
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    labels = latent_dataframe["labels"]
    num_classes = len(labels.unique())

    concepts_original = latent_dataframe.iloc[:, 2 : num_concepts + 2]
    c_samples_original = torch.tensor(
        concepts_original.to_numpy(), dtype=torch.float32
    ).to(device)

    side_channel_original = latent_dataframe.iloc[:, num_concepts + 2 :]
    z_samples_original = torch.tensor(
        side_channel_original.to_numpy(), dtype=torch.float32
    ).to(device)

    task_accuracies_random_side_channel = []
    task_accuracies_random_concepts = []

    for _ in range(repeat):
        concepts = concepts_original.apply(
            lambda col: col.sample(frac=1).reset_index(drop=True)
        )
        c_samples = torch.tensor(concepts.to_numpy(), dtype=torch.float32).to(device)

        side_channel = side_channel_original.apply(
            lambda col: col.sample(frac=1).reset_index(drop=True)
        )
        z_samples = torch.tensor(side_channel.to_numpy(), dtype=torch.float32).to(
            device
        )

        logits_side_channel = model(
            torch.cat([c_samples_original, z_samples], dim=1).to(device)
        )

        logits_concepts = model(
            torch.cat([c_samples, z_samples_original], dim=1).to(device)
        )

        if num_classes == 2:
            task_probs = torch.sigmoid(logits_side_channel)
            task_preds = (task_probs > 0.5).int()
            task_acc_side_channel = accuracy_score(task_preds.tolist(), labels)

            task_probs = torch.sigmoid(logits_concepts)
            task_preds = (task_probs > 0.5).int()
            task_acc_concepts = accuracy_score(task_preds.tolist(), labels)

        else:
            task_preds = torch.argmax(logits_side_channel, dim=1)
            task_acc_side_channel = accuracy_score(task_preds.tolist(), labels)

            task_preds = torch.argmax(logits_concepts, dim=1)
            task_acc_concepts = accuracy_score(task_preds.tolist(), labels)

        task_accuracies_random_side_channel.append(task_acc_side_channel)
        task_accuracies_random_concepts.append(task_acc_concepts)

    """
        task_accuracies_random_side_channel - overall accuracy with randomised side channel
        task_accuracies_random_concepts - overall accuracy with randomised concepts
    """

    return np.mean(task_accuracies_random_concepts), np.mean(
        task_accuracies_random_side_channel
    )
