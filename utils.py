import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
from typing import List
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from transformers import CLIPVisionModelWithProjection
from torch.utils.data import DataLoader
from transformers import logging as hf_logging
from transformers import CLIPImageProcessor



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
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")

    print(f"\n{"="*60}\nResults:    \n{"="*60}")
    print(f"Accuracy:               {accuracy:.4f}")
    print(f"Precision (macro):      {precision:.4f}")
    print(f"Recall (macro):         {recall:.4f}")
    print(f"F1 (macro):             {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def plot_conf_matrix(y_true: List, y_pred: List, title: str ="Confusion Matrix", normalize=True, display_labels=None):
    """Plots a confusion matrix for a set of preditions and true values with a configurable title.
    
    Parameters
    ----------
    y_true : List
        List containing the true labels
    y_pred : List
        List containing the predicted labels
    title : str (optional)
        Title for the confusion matrix figure, defaults to "Confusion Matrix".
    normalize : bool, default True
        Whether or not to normalize the values in the matrix.
    display_labels (optional)
        The list of labels to be used in the Confusion Matrix figure.
    """

    # handle normalization parameter
    if normalize:
        normalize = "true"
    else:
        normalize = None

    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    cm_rounded = np.round(cm, 2)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_rounded, display_labels=display_labels)

    fig, ax = plt.subplots(figsize=(8,8))
    disp.plot(ax=ax, xticks_rotation=45)

    plt.title(title)
    plt.grid(False)
    plt.show()

def get_device():
    """Return gpu if available, otherwise return cpu. Handles cases for both Windows (cuda) and Apple Mac (mps)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def extract_image_embeddings(dataset, device):
    """Extracts CLIP image embeddings and class labels from a given dataset.
    Requirement: images have been preprocessed into pixel values with CLIPImageProcessor.
    
    Parameters
    ----------
    dataset
        The dataset from which to compute the image embeddings
    device
        The device which handles the model inference computations, preferably a gpu.

    Returns
    -------
    Numpy.Array
        A numpy array of shape (dataset_size, 512) containing the image embeddings.
    List[str]
        A list containing the class labels of the dataset.
    """
    vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)

    # create DataLoader for processing in batches
    dataset.set_format("torch", columns=["pixel_values"], output_all_columns=True)
    loader = DataLoader(dataset, batch_size=32, num_workers=4)

    all_embeds = []
    all_labels = []

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        batch_labels = batch["label"].tolist()
        with torch.no_grad():
            image_embeds = vision_model(pixel_values).image_embeds
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        all_labels.extend(batch_labels)
        all_embeds.append(image_embeds)

    return torch.cat(all_embeds).cpu().numpy(), all_labels

def sample_few_shot_embeddings(k: int, seed: int, dataset, device = None):
    """Sample k images per class from a dataset and compute their CLIP image embeddings.

    For each class label present in `dataset`, randomly selects k row indices
    (without replacement) using a numpy random seed, gathers the resulting
    k-shot subset, preprocesses the images into pixel values, and computes
    CLIP image embeddings for the sampled subset.

    Args:
        k (int): Number of images to sample per class. Must not exceed the available number of images in smallest class
        seed (int): Seed for `numpy.random`, applied before each class's sampling
            call to make the selection reproducible.
        dataset: A dataset with a `"label"` column and a `features["label"].names`
            attribute listing class names (e.g. a Hugging Face `datasets.Dataset`).
        device (optional): Device (e.g. "cpu" or "cuda") used for computing embeddings.
            If None, defaults to the device returned by `get_device()`.

    Returns:
        tuple:
            k_image_embeddings (np.ndarray): CLIP image embeddings for the sampled
                k-shot subset, shape (k * num_classes, embedding_dim).
            k_sample_labels (np.ndarray or list): Class labels corresponding to
                each sampled embedding.
    """

    # Use gpu, if available, as default device
    if not device:
        device = get_device()
    
    # class labels present in dataset
    labels = dataset.features["label"].names
    
    # Randomly select k images from each class
    all_idx = []                                                    # list to collect all selected row indeces
    for class_id in range(len(labels)):                             # loop through each class_id
        label_col = np.array(dataset["label"])                      # extract label column as np.array
        idx = np.where(label_col == class_id)[0]                    # extract all indexes of rows of class_id
        np.random.seed(seed)                                        # set seed for numpy.random
        k_idx = np.random.choice(idx, k, replace=False).tolist()    # choose k random indeces for current class
        all_idx.extend(k_idx)                                       # append chosen indeces
    k_sample = dataset.select(all_idx)                              # select randomly smapled indeces from dataset

    # compute image embeddings for k-sample of training images

    # preprocessing function to extract pixel values from images 
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    def preprocess_images(examples):
        images = examples["image"]
        inputs = image_processor(images, return_tensors="pt")
        examples["pixel_values"] = inputs["pixel_values"]
        return examples

    k_sample_processed = k_sample.map(preprocess_images, batched=True, remove_columns=["image"])    # extract pixel values
    k_image_embeddings, k_sample_labels = extract_image_embeddings(k_sample_processed, device)      # compute CLIP image embeddings

    return k_image_embeddings, k_sample_labels
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

    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    def preprocess_images(examples):
            images = examples["image"]
            inputs = image_processor(images, return_tensors="pt")
            examples["pixel_values"] = inputs["pixel_values"]
            return examples

    dataset_processed = dataset.map(preprocess_images, batched = True, remove_columns =["image"])
    dataset_processed.set_format("torch", columns = ["pixel_values"], output_all_columns = True)
    loader = DataLoader(dataset_processed, batch_size = 32, num_workers = 4)

    all_embeds = []

    device = get_device()

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

