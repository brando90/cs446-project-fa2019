import unittest

import utils.utils as utils

from pdb import set_trace as st

class TestStringMethods(unittest.TestCase):

    def test_imports_through_packages(self):
        helloworld = utils.HelloWorld()
        self.assertTrue( helloworld == 'HelloWorld')


if __name__ == '__main__':
    unittest.main()
