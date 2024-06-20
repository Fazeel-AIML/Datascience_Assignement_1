import torch

def perform_inference(model, sentence, word_to_index, tag_to_index, device):
    sentence_tensor = torch.tensor([word_to_index.get(word, word_to_index["<unk>"]) for word in sentence.split(" ")]).to(device)
    model.eval()
    with torch.no_grad():
        output = model(sentence_tensor)
    predicted_class = output.argmax().item()
    return {v: k for k, v in tag_to_index.items()}.get(predicted_class, "Tag not found")
