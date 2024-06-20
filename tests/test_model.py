import unittest
import torch
from src.model import BoW

class TestModel(unittest.TestCase):
    def test_bow_forward(self):
        model = BoW(10, 2)
        input_tensor = torch.LongTensor([1, 2, 3])
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 2))

if __name__ == '__main__':
    unittest.main()
