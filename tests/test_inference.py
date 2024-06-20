import unittest
import torch
from src.model import BoW
from src.inference import perform_inference

class TestInference(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.word_to_index = {"<unk>": 0, "i": 1, "love": 2, "programming": 3}
        self.tag_to_index = {"positive": 0, "negative": 1}
        self.model = BoW(len(self.word_to_index), len(self.tag_to_index)).to(self.device)
        self.model.eval()

    def test_perform_inference(self):
        sentence = "i love programming"
        predicted_tag = perform_inference(self.model, sentence, self.word_to_index, self.tag_to_index, self.device)
        self.assertIn(predicted_tag, self.tag_to_index)

if __name__ == '__main__':
    unittest.main()