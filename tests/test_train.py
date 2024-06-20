import unittest
import torch
from src.model import BoW, create_tensors
from src.train import train_bow

class TestTrain(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.word_to_index = {"<unk>": 0, "i": 1, "love": 2, "dogs": 3}
        self.tag_to_index = {"positive": 0, "negative": 1}
        self.train_data = [(["i", "love", "dogs"], "positive")]
        self.train_tensors = list(create_tensors(self.train_data, self.word_to_index, self.tag_to_index))
        self.model = BoW(len(self.word_to_index), len(self.tag_to_index)).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = torch.nn.CrossEntropyLoss()

    def test_train_bow(self):
        train_bow(self.model, self.optimizer, self.criterion, self.train_tensors, self.train_tensors, self.device, num_iters=1)
        self.assertTrue(self.model.training is False)  # Ensures model is in eval mode after training

if __name__ == '__main__':
    unittest.main()
