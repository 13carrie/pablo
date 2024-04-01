# macOS command for running this test suite while cwd = code:
# python3 -m unittest Scripts/test_split_data.py

import unittest
from pyspark.sql import SparkSession
from Scripts.split_data import split_dataset, match_keyword, isolate_attacks
class TestClass(unittest.TestCase):

    def setUp(self):
        spark = SparkSession.builder.appName("Pablo split test 1") \
            .config("spark.driver.memory", "15g") \
            .getOrCreate()
        self.df = spark.createDataFrame([('1', 'PortScan'), ('2', 'FTP-Patator'), ('3', 'DDoS'),
                                         ('4', 'PortScan'), ('5', 'SSH-Patator'), ('6', 'DDoS'),
                                         ('7', 'PortScan'), ('8', 'Hulk'), ('9', 'Hulk'), ('10', 'Hulk')],
                                        ['id', 'Label'])

    def test_split_dataset(self):
        # if args <=2, random split should result in both dfs containing rows (since this is a very small dataframe,
        # randomsplit does not provide an accurate train:test split)
        training_data, test_data = split_dataset(self.df, [])
        self.assertTrue((training_data.count() + test_data.count()) <= 10)
        self.assertTrue((training_data.count() + test_data.count()) > 0)

        # if args >2, train and test datasets should equal match keyword train and test datasets
        training_data, test_data = split_dataset(self.df, ['1', '2', 'pbp'])
        training_data_2, test_data_2 = match_keyword(self.df, 'pbp')
        self.assertEqual(training_data.collect(), training_data_2.collect())
        self.assertEqual(training_data.schema, training_data_2.schema)

        # if case for a keyword is not implemented, randomsplit should occur
        training_data, test_data = split_dataset(self.df, ['1', '2', 'test'])
        training_data_2, test_data_2 = match_keyword(self.df, 'test')
        self.assertTrue((training_data.count() + test_data.count()) <= 10)
        self.assertTrue((training_data.count() + test_data.count()) > 0)
        self.assertTrue((training_data_2.count() + test_data_2.count()) <= 10)
        self.assertTrue((training_data_2.count() + test_data_2.count()) > 0)


    def test_match_keyword(self):
        # if keyword = pbp
        training_data, test_data = match_keyword(self.df, 'pbp')
        self.assertFalse(training_data.isEmpty())
        self.assertFalse(test_data.isEmpty())
        self.assertTrue(training_data.count() == 5)
        self.assertTrue(test_data.count() == 5)
        self.assertTrue(training_data.filter(training_data.Label == 'PortScan').isEmpty())
        self.assertTrue(training_data.filter(training_data.Label == 'FTP-Patator').isEmpty())
        self.assertTrue(training_data.filter(training_data.Label == 'SSH-Patator').isEmpty())
        self.assertTrue(test_data.filter(test_data.Label == 'PortScan').count() == 3)
        self.assertTrue(test_data.filter(test_data.Label == 'FTP-Patator').count() == 1)
        self.assertTrue(test_data.filter(test_data.Label == 'SSH-Patator').count() == 1)

        # if keyword = not-pbp
        training_data, test_data = match_keyword(self.df, 'not-pbp')
        self.assertTrue((training_data.count() + test_data.count()) <= 10)
        self.assertTrue((training_data.count() + test_data.count()) > 0)

        # if keyword is empty string, random split should occur and both shouldn't be empty
        training_data, test_data = match_keyword(self.df, '')
        self.assertTrue((training_data.count() + test_data.count()) <= 10)
        self.assertTrue((training_data.count() + test_data.count()) > 0)

    def test_isolate_attacks(self):
        # if there is one attack specified and present in df
        remaining_df, isolated_df = isolate_attacks(self.df, ['PortScan'])
        self.assertFalse(isolated_df.isEmpty())
        self.assertTrue(remaining_df.where(remaining_df.Label == 'PortScan').isEmpty())
        self.assertFalse(isolated_df.where(isolated_df.Label == 'PortScan').isEmpty())
        self.assertTrue(isolated_df.count() == 3)
        self.assertTrue(remaining_df.count() == 7)

        # if there is more than one attack specified and present in df
        remaining_df, isolated_df = isolate_attacks(self.df, ['PortScan', 'FTP-Patator'])
        self.assertFalse(isolated_df.isEmpty())
        self.assertTrue(remaining_df.where(remaining_df.Label == 'PortScan').isEmpty())
        self.assertTrue(remaining_df.where(remaining_df.Label == 'FTP-Patator').isEmpty())
        self.assertFalse(isolated_df.where(isolated_df.Label == 'PortScan').isEmpty())
        self.assertFalse(isolated_df.where(isolated_df.Label == 'FTP-Patator').isEmpty())
        self.assertTrue(isolated_df.count() == 4)
        self.assertTrue(remaining_df.count() == 6)

        # if there are attacks specified that are not present in df
        remaining_df, isolated_df = isolate_attacks(self.df, ['Other Attack'])
        self.assertTrue(isolated_df.isEmpty())
        self.assertEqual(self.df, remaining_df)
        self.assertTrue(isolated_df.count() == 0)
        self.assertTrue(remaining_df.count() == 10)

        # if empty list is passed
        remaining_df, isolated_df = isolate_attacks(self.df, [])
        self.assertTrue(isolated_df.isEmpty())
        self.assertEqual(self.df, remaining_df)
        self.assertTrue(isolated_df.count() == 0)
        self.assertTrue(remaining_df.count() == 10)

    def tearDown(self):
        self.df.unpersist()
