import shelve
from datetime import date
import string
import numpy as np

# fpath = 'test.p'
# master = shelve.open(fpath, writeback=True)

# keys = string.ascii_lowercase
# data = dict.fromkeys(keys[:5], np.random.randint(10))

# new_ver = 2

# # if file is not empty
# if 'ver' in master:
#     current_ver = master['ver']

#     # if master is up-to-date
#     if current_ver == new_ver:
#         master['data'] = data
        
#         # read from master
#         print(len(master.keys()))

#     # else if current version is outdated, query new data
#     else:
#         master['ver'] = new_ver
#         print('updated master')
        
# # if file is empty
# else: 
#     master['ver'] = 1
#     print('write new to master')
#     master['data'] = data

# master.close()

punc = string.punctuation
print(punc)
print(punc.replace("|", ""))


