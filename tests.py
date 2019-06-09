import unittest


from text_analysis import TextAnalysis


class TestChallenge(unittest.TestCase):
    def test_check_time(self):
        result = 1
        text = "J'ai 3ans "
        self.assertEqual(TextAnalysis.check_time(text), result)

        text = "J'ai 1 an"
        self.assertEqual(TextAnalysis.check_time(text), result)

        text = "J'ai 1 mois."
        self.assertEqual(TextAnalysis.check_time(text), result)

        text = "J'ai 2 an ."
        self.assertEqual(TextAnalysis.check_time(text), result)

        text = "il est 4 Heure."
        self.assertEqual(TextAnalysis.check_time(text), result)

        text = "J'ai  le 5 août ."
        self.assertEqual(TextAnalysis.check_time(text), result)

        text = "J'ai 4 janvier ."
        self.assertEqual(TextAnalysis.check_time(text), result)

        text = "J'ai 4 fevrier ."
        self.assertEqual(TextAnalysis.check_time(text), result)

        text = "J'ai 2 heures "
        self.assertEqual(TextAnalysis.check_time(text), result)

        text = "J'ai depuis lundi"
        self.assertEqual(TextAnalysis.check_time(text), result)

        text = "A 3 heure20"
        self.assertEqual(TextAnalysis.check_time(text), result)

        text = "J'ai 4min ."
        self.assertEqual(TextAnalysis.check_time(text), result)

        text = "J'ai 4s."
        self.assertEqual(TextAnalysis.check_time(text), result)

        text = "J'ai 235sec"
        self.assertEqual(TextAnalysis.check_time(text), result)

        text = "Lundi j ai"
        self.assertEqual(TextAnalysis.check_time(text), result)

        # False
        result = 0
        text = "J'aime ansible"
        self.assertEqual(TextAnalysis.check_time(text), result)

        text = "J'aime milan"
        self.assertEqual(TextAnalysis.check_time(text), result)

        text = "J'aime le mans."
        self.assertEqual(TextAnalysis.check_time(text), result)

        text = "J'aime mon moi moisi "
        self.assertEqual(TextAnalysis.check_time(text), result)


if __name__ == "__main__":
    unittest.main()
