import random
import torch

def train_bow(model, optimizer, criterion, train_data, test_data, device, num_iters=10):
    for epoch in range(num_iters):
        model.train()
        random.shuffle(train_data)
        total_loss = 0.0
        train_correct = 0
        
        for sentence, tag in train_data:
            sentence = torch.tensor(sentence).to(device)
            tag = torch.tensor([tag]).to(device)
            optimizer.zero_grad()
            output = model(sentence)
            loss = criterion(output, tag)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_correct += (output.argmax(1) == tag).sum().item()
        
        model.eval()
        test_correct = sum((model(torch.tensor(sentence).to(device)).argmax(1) == tag).sum().item() for sentence, tag in test_data)
        
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_data)}, Train Accuracy: {train_correct/len(train_data)}, Test Accuracy: {test_correct/len(test_data)}')
