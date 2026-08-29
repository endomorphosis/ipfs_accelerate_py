from unittest import TestCase

from demo.labels import label


class TestLabel(TestCase):
    def test_label(self) -> None:
        self.assertEqual(label(7), "value:7")
