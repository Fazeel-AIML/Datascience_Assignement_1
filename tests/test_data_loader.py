import unittest
from src.data_loader import read_data

class TestDataLoader(unittest.TestCase):
    def test_read_data(self):
        data = read_data('data/classes/test.txt')
        self.assertTrue(len(data) > 0)
        self.assertEqual(len(data[0]), 2)

if __name__ == '__main__':
    unittest.main()
