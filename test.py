import unittest
import requests
from sklearn.metrics import accuracy_score


class My_sentiment_app(unittest.TestCase):
    def test_text(self):
        """we suppose that the text in contained in a variable my_text and we figue if the text is not empty and is a string"""
        my_text=requests.get("http://127.0.0.1:5000/?comments=writing+text&submit=Submit")

        self.assertEqual(my_text.status_code,200)
        


    def test_model(self):
        "we test if our model has a better acccuracy than 80 percent"

        result= open('accuracy.txt', 'r')
        content = result.read()
        self.assertTrue(float(content) >0.80)



if __name__ == "__main__":
    unittest.main()


#python -m unittest tests/test_something.py
