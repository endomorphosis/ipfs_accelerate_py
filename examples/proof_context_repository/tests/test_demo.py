from unittest import TestCase

from demo import increment


class TestIncrement(TestCase):
    def test_increment(self) -> None:
        self.assertEqual(increment(2), 3)
