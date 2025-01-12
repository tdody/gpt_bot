"""
A module for handling the dataset.
"""

from torch.utils.data import Dataset
from unidecode import unidecode


class ContentDataset(Dataset):
    words: list[str]
    index_to_token: dict[int, str]
    token_to_index: dict[str, int]
    max_word_length: int

    def __init__(self, words: list[str], tokens: list[str], max_word_length: int):
        self.words = words
        self.index_to_token = {i: token for i, token in enumerate(tokens)}
        self.token_to_index = {token: i for i, token in enumerate(tokens)}
        # note: we do not simply retrieve the max length of the words
        # because we use ContentDataset both on the training and testing
        # phases, and we need to ensure that the model is able to handle
        # the same word length on both phases
        max_word_length = max_word_length


def __clean_word(word: str) -> str:
    # Remove leading and trailing whitespaces
    # Convert to lowercase
    # Remove accents
    # Remove special characters
    # Replace spaces with underscores

    word = word.strip().lower()
    word = word.replace(" ", "_")
    word = unidecode(word)
    return word


def create_datasets(file_path: str) -> tuple[ContentDataset, ContentDataset]:
    """ "
    Create the training and testing datasets.

    Args:
        file_path (str): The path to the file containing the dataset.

    Returns:
        tuple[ContentDataset, ContentDataset]: The training and testing datasets.
    """
    with open(file_path, "r") as file:
        content = file.read()

    # Split the content into words and normalize them
    words = content.splitlines()
    words = [__clean_word(word) for word in words]
    words = [w for w in words if w]

    tokens = sorted(list(set("".join(words))))
    max_context = max([len(w) for w in words])
