from rdblearn.datasets import RDBDataset

dataset = RDBDataset.from_relbench("rel-avito")
rdb = dataset.rdb
print(rdb)
print(rdb.tables.keys())