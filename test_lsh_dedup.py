import csv
import itertools
import unittest

from lsh_dedup import MinHashLSH, DeduplicationIndex

TEST_FILE = "test_resources/texts.csv"

def _read_csv_records():
    with open(TEST_FILE, "r") as src:
        return list(row[0] for row in csv.reader(src))

class LSHDedupTestCase(unittest.TestCase):
    THRESHOLD = 0.5
    STRICT_THRESHOLD = 1.0
    RECORDS = _read_csv_records()
    LSH = MinHashLSH(RECORDS, 8, 1)
    MINHASHES = LSH.get_minhash_index()
    DEDUP_INDEX = DeduplicationIndex(LSH, threshold=THRESHOLD)
    GROUPED_INDICES = DEDUP_INDEX.grouped_indices()

    def test_hash_count(self):
        self.assertEqual(len(self.MINHASHES), len(self.RECORDS))

    def test_lsh_query(self):
        for idx, minhash in self.MINHASHES.items():
            self.assertIn(idx, self.LSH.query(minhash))

    def test_lsh_query_threshold(self):
        for idx, minhash in self.MINHASHES.items():
            result = self.LSH.query(minhash, self.THRESHOLD)
            self.assertIn(idx, result)
            for res_idx in result:
                res_hash = self.MINHASHES[res_idx]
                self.assertGreaterEqual(minhash.jaccard_similarity(res_hash), self.THRESHOLD)

    def test_lsh_query_threshold_strict(self):
        for idx, minhash in self.MINHASHES.items():
            result = self.LSH.query(minhash, self.STRICT_THRESHOLD)
            self.assertIn(idx, result)
            for res_idx in result:
                res_hash = self.MINHASHES[res_idx]
                self.assertEqual(minhash.jaccard_similarity(res_hash), self.STRICT_THRESHOLD)

    def test_dedup_index_groups_ct(self):
        self.assertGreater(len(self.GROUPED_INDICES), 2)

    def test_dedup_index_union(self):
        groups_union = set()
        for grp in self.GROUPED_INDICES:
            groups_union |= set(grp)

    def test_dedup_index_intersection(self):
        for ref, cmp  in itertools.combinations_with_replacement(self.GROUPED_INDICES, 2):
            if ref != cmp:
                assert not set(ref).intersection(cmp)
