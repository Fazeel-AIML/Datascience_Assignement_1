import torch
from src.data_loader import download_data, read_data
from src.model import BoW, create_tensors
from src.train import train_bow
from src.inference import perform_inference

urls = [
    "https://raw.githubusercontent.com/neubig/nn4nlp-code/master/data/classes/dev.txt",
    "https://raw.githubusercontent.com/neubig/nn4nlp-code/master/data/classes/test.txt",
    "https://raw.githubusercontent.com/neubig/nn4nlp-code/master/data/classes/train.txt"
]

download_data(urls)
train_data = read_data('data/classes/train.txt')
test_data = read_data('data/classes/test.txt')

word_to_index = {"<unk>": 0}
tag_to_index = {}
for data in train_data:
    for word in data[1].split(" "):
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)
    if data[0] not in tag_to_index:
        tag_to_index[data[0]] = len(tag_to_index)

train_tensors = list(create_tensors(train_data, word_to_index, tag_to_index))
test_tensors = list(create_tensors(test_data, word_to_index, tag_to_index))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BoW(len(word_to_index), len(tag_to_index)).to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()

train_bow(model, optimizer, criterion, train_tensors, test_tensors, device)

sentence = "I love programming"
predicted_tag = perform_inference(model, sentence, word_to_index, tag_to_index, device)
print(f"Predicted Tag: {predicted_tag}")
