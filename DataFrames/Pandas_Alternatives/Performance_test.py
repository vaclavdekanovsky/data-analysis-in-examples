import sys
import os
import gc
import psutil
from time import time, sleep, strftime, localtime
import pandas as pd
import dask.dataframe as dd
#import modin.pandas as mpd
import vaex
from pyspark.sql import SparkSession, functions

# data based on https://www.kaggle.com/c/ieee-fraud-detection/data
folder = "/home/vaclav/Data/Kaggle/EEE-CIS_Fraud_Detection"
files = ["train_transaction.csv", "train_identity.csv"]
paths = [os.path.join(folder, f) for f in files]

class Events:
    
    def __init__(self, path):
        self.file = open(path, 'a', encoding='utf-8')
        
    def log(self, time_end, tool, operation, duration):
        self.file.write("|".join([strftime('%Y-%m-%d %H:%M:%S', localtime(time_end)),tool,operation,str(duration)])+"\n")
        
    def close(self):
        self.file.close()

def run_pandas(logger):
	"""Performance test in pandas"""

	s = {}

	ts = time()
	df = pd.read_csv(paths[0])
	te = time()
	s["load_transactions"] = te-ts
	logger.log(te, "pandas", "load_transactions",  te-ts)

	ts = time()
	df2 = pd.read_csv(paths[1])
	te = time()
	s["load_identity"] = te-ts
	logger.log(te, "pandas", "load_identity",  te-ts)

	ts = time()
	dff = df.merge(df2, on="TransactionID")
	te = time()
	s["merge"] = te-ts
	logger.log(te, "pandas", "merge",  te-ts)

	ts = time()
	grp = dff.groupby(["isFraud","ProductCD","card4","card6","id_15","id_31"])["TransactionAmt"].agg(["mean","sum"])
	te = time()
	s["aggregation"] = te-ts
	logger.log(te, "pandas", "aggregation",  te-ts)

	ts = time()
	dff.sort_values(by=["card1","addr1","D9"], inplace=True)
	dff.sort_values(by=["addr1","D9","card1"], inplace=True)
	dff.sort_values(by=["D9","card1","addr1"], inplace=True)
	te = time()
	s["sorting"] = te-ts
	logger.log(te, "pandas", "sorting",  te-ts)
	
	return s

def run_dask(logger):
	s = {}

	ts = time()
	df = dd.read_csv(paths[0])
	te = time()
	s["load_transactions"] = te-ts
	logger.log(te, "dask", "load_transactions",  te-ts)

	ts = time()
	df2 = dd.read_csv(paths[1])
	te = time()
	s["load_identity"] = te-ts
	logger.log(te, "dask", "load_identity",  te-ts)
	ts = time()
	dff = df.merge(df2, on="TransactionID")
	te = time()
	s["merge"] = te-ts
	logger.log(te, "dask", "merge",  te-ts)

	# the difference is that we call compute method, which runs all the computations at this point
	ts = time()
	grp = dff.groupby(["isFraud","ProductCD","card4","card6","id_15","id_31"])["TransactionAmt"]\
		.agg(["mean","sum"])\
		.compute()
	te = time()
	s["aggregation"] = te-ts
	logger.log(te, "dask", "aggregation",  te-ts)

	# parallel soring is tricky that is why there are only work arounds in dask. 
	ts = time()
	dff.set_index("card1").compute()
	te = time()
	s["sorting"] = te-ts
	logger.log(te, "dask", "sorting",  te-ts)

def run_vaex(logger):
	s = {}

	ts = time()
	df = vaex.open(paths[0])
	te = time()
	s["load_transactions"] = te-ts
	logger.log(te, "vaex", "load_transactions",  te-ts)

	ts = time()
	df2 = vaex.open(paths[1])
	te = time()
	s["load_identity"] = te-ts
	logger.log(te, "vaex", "load_identity",  te-ts)

	ts = time()
	dff = df.join(df2, on="TransactionID")
	te = time()
	s["merge"] = te-ts	
	logger.log(te, "vaex", "merge",  te-ts)

	# the difference is that we call compute method, which runs all the computations at this point
	ts = time()
	grp = dff.groupby([dff["isFraud"],dff["ProductCD"],dff["card4"],dff["card6"],dff["id_15"],dff["id_31"]], 
					agg=[vaex.agg.mean('TransactionAmt'), vaex.agg.sum('TransactionAmt')])
	te = time()
	s["aggregation"] = te-ts
	logger.log(te, "vaex", "aggregation",  te-ts)

	# the difference is that we call compute method, which runs all the computations at this point
	ts = time()
	dff_s = dff.sort(by=["card1","addr1","D9"])
	dff_s = dff.sort(by=["addr1","D9","card1"])
	dff_s = dff.sort(by=["D9","card1","addr1"])
	te = time()
	s["sorting"] = te-ts
	logger.log(te, "vaex", "sorting",  te-ts)

def run_spark(my_spark, logger):
	s = {}

	tool = "spark"
	s = {}

	ts = time()
	df = my_spark.read.csv(paths[0],inferSchema = True,header= True) 
	te = time()
	s["load_transactions"] = te-ts
	logger.log(te, "spark", "load_transactions",  te-ts)

	ts = time()
	df2 = my_spark.read.csv(paths[1],inferSchema = True,header= True) 
	te = time()
	s["load_identity"] = te-ts
	logger.log(te, "spark", "load_identity",  te-ts)

	ts = time()
	dff = df.join(df2, "TransactionID")
	te = time()
	s["merge"] = te-ts
	logger.log(te, "spark", "merge",  te-ts)

	# the difference is that we call collect method, which runs all the computations at this point
	ts = time()
	grp = dff.groupby(["isFraud","ProductCD","card4","card6","id_15","id_31"]) \
			.agg(functions.avg("TransactionAmt"), functions.sum("TransactionAmt"))\
			.collect()
	te = time()
	s["aggregation"] = te-ts
	logger.log(te, "spark", "aggregation",  te-ts)

	ts = time()
	dff.orderBy("card1","addr1","D9").collect()
	# alternatively
	# dff.sort("card1","addr1","D9").collect()
	te = time()
	s["sorting"] = te-ts
	logger.log(te, "spark", "sorting",  te-ts)

def run_modin(logger):
	s = {}

	ts = time()
	df = mpd.read_csv(paths[0])
	te = time()
	s["load_transactions"] = te-ts
	logger.log(te, "modin", "load_transactions",  te-ts)

	ts = time()
	df2 = mpd.read_csv(paths[1])
	te = time()
	s["load_identity"] = te-ts
	logger.log(te, "modin", "load_identity",  te-ts)

	ts = time()
	dff = df.merge(df2, on="TransactionID")
	te = time()
	s["merge"] = te-ts
	logger.log(te, "modin", "merge",  te-ts)

	# modin defaults to pandas for multiple column aggregation and then fails on KeyError, though the key is available
	ts = time()
	try:
		grp = dff.groupby(["isFraud","ProductCD","card4","card6","id_15","id_31"])["TransactionAmt"].agg(["mean","sum"])
	except Exception as e:
		print(e)
	te = time()
	s["aggregation"] = te-ts
	logger.log(te, "modin", "aggregation",  te-ts)

def system_resources(n, pause, cpu_threshold = 0.5, mem_threshold = 0.5):
    
    cpu = []
    mem = []
    cpu_within_limit = True
    mem_within_limit = True
    
    for i in range(n):
        cpu.append(psutil.cpu_percent())
        mem.append(psutil.virtual_memory().percent)
        sleep(pause)
    cpu = sum(cpu)/n
    mem = sum(mem)/n
    
    if cpu / 100 > cpu_threshold:
        cpu_within_limit = False
        
    if mem / 100 > mem_threshold:
        mem_within_limit = False
    
    return {"cpu": cpu, "memory": mem, "cpu_limit": cpu_within_limit, "mem_limit": mem_within_limit }
    
def clean(wait_time: int=15):
    """Cleans created DataFrames and call the garbage collector to actions. Wait for 15s by default"""
    df, df2, dff, grp = None, None, None, None
    gc.collect()
    sleep(wait_time)
    return None

def check_resources():
	# if memory or cpu usage is high, clean resources and wait 60s
	res_breached = True
	while res_breached:
		res = system_resources(3,1)
		if res["mem_limit"] and res["cpu_limit"]:
			res_breached = False
		else:
			print(f"CPU/Memory over limit {res}")
			clean(60)
    
if __name__ == "__main__":
	
	# logging 
	logger = Events("l_2.log")

	# Create my_spark
	my_spark = SparkSession.builder \
		.master("local") \
		.appName("Pandas Alternative") \
		.config("spark.some.config.option", "some-value") \
		.getOrCreate()
	
	# 7 rounds
	for i in range(7):

		check_resources()

		print(f"{i} pandas")
		run_pandas(logger)
		check_resources()

		print(f"{i} vaex")
		run_vaex(logger)
		check_resources()

		print(f"{i} dask")
		run_dask(logger)
		check_resources()

		print(f"{i} spark")
		run_spark(my_spark, logger)
		check_resources()

		#run_modin(logger)
		#check_resources()
	
	logger.close()
	my_spark.stop()