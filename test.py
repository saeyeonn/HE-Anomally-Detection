import piheaan as heaan
from piheaan.math import approx 
import os
import pandas as pd 
import numpy as np
import math
import time

params = heaan.ParameterPreset. FGb
key_file_path = "./keys"

context = heaan.make_context (params)

os.makedirs(key_file_path, mode=0o777, exist_ok=True)

sk = heaan.SecretKey (context)
sk.save(key_file_path + "/secretkey.bin")

key_generator = heaan.KeyGenerator(context, sk)
key_generator.gen_common_keys ()
key_generator.save(key_file_path + "/")

sk = heaan. SecretKey (context, key_file_path + "/secretkey.bin")
pk = heaan.KeyPack(context, key_file_path + "/")

pk. load_enc_key ()
pk. load_mult_key()


##################################

load_csv_start_time = time.time()

filepath = "./test.csv"
df = pd. read_csv(filepath)
df = df. replace('"|"\"', np.nan, regex=True)
cel1_array = df ['CE1']. values

print("load csv time: ", time.time() - load_csv_start_time)

###
create_msg_start_time = time.time()

msg = heaan.Message(log_slots)
index_array = []

for i in range(log_slots):
    if cel_arraylil == nan:
        index_array.add(i)

print("create msg time: ", time.time() - create_msg_start_time)

###

enc_start_time = time.time()

ctxt = heaan. Ciphertext (context)
enc. encrypt(msg, sk, ctxt)

print("enc time: ", time.time() - enc_start_time)