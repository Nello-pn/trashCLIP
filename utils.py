import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from typing import List
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    ConfusionMatrixDisplay
)

def count_observations(dataset):
    """
    Counts the number of observations for each label in the given dataset.

    Parameters
    ----------
    dataset :datasets.arrow_dataset.Dataset
        The dataset to count observations from.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the count of observations for each label.
    """
    labels = dataset.features["label"].names
    label_counts = {label: dataset["label"].count(labels.index(label)) for label in labels}
    label_counts = pd.DataFrame.from_dict(label_counts, orient="index", columns=["count"]).reset_index(names="label")
    return label_counts

def evaluate_predictions(y_true : List, y_pred : List, target_names=None):
    """Calculates and prints classification evaluation metrics accuracy, precision, recall, and f1-score and prints the classification report.
    
    Parameters
    ----------
    y_true : List
        List containing the true labels
    y_pred : List
        List containing the predicted labels
    target_names : List
        List of label names to be displayed in the classification report. If None, defaults to the variable "labels" if defined.

    Returns
    -------
    dict
        A dictionary with keys "accuracy", "precision", "recall", and "f1", containing the respective metric values
    """
    if not target_names:
        try: target_names = labels  # type: ignore
        except NameError: pass

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

    print(f"\n{"="*60}\nResults:    \n {"="*60}")
    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1:             {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def plot_conf_matrix(y_true: List, y_pred: List, title: str ="Confusion Matrix"):
    """Plots a confusion matrix for a set of preditions and true values with a configurable title.
    
    Parameters
    ----------
    y_true : List
        List containing the true labels
    y_pred : List
        List containing the predicted labels
    title : str (optional)
        Title for the confusion matrix figure, defaults to "Confusion Matrix".
    """

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, xticks_rotation=45)
    plt.title(title)
    plt.grid(False)
    plt.show()

def embedding_dataframe_construction(dataset, labels, embedding_model):
    """Builds a Pandas Dataframe using the computed Image Embeddings Only
    
    Parameters
    ----------
    dataset : List
        List of images being processed 
    labels : List
        List of the labels of the images to add to the 'Class' column
    embedding_model : Pytorch neural network model (CLIPVisionModelWithProjection)
        CLIPVisionModelWithProjection model for extracting image embeddings.
        
    return
    -----------
    data_frame: pd.Dataframe
        Pandas dataframe of shape (nX512) containing the embeddings and the target value
    """

    dataset_processed = dataset.map(preprocess_images, batched = True, remove_columns =["image"])
    dataset_processed.set_format("torch", columns = ["pixel_values"], output_all_columns = True)
    loader = DataLoader(dataset_processed, batch_size = 32, num_workers = 4)

    all_embeds = []

    for batch in tqdm(loader):
        with torch.no_grad():
            inputs = batch["pixel_values"].to(device)
            outputs = embedding_model(inputs)
        image_embeds = outputs.image_embeds
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    
        all_embeds.append(image_embeds)
    
    image_embeddings = torch.cat(all_embeds)
    list_embeds = image_embeddings.cpu().numpy()
    data_frame = pd.DataFrame(list_embeds)
    data_frame['Class'] = labels

    return data_frame

