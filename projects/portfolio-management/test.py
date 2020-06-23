import shelve
from datetime import date

fpath = 'master.p'
master = shelve.open(fpath, writeback=True)

# # if file is not empty
# if 'ver' in master:
#     current_ver = master['ver']

#     # if master is up-to-date
#     if current_ver == 2:
#         print('reading from today\'s version')
        
#         # read from master
#         print(len(master.keys()))

#     # else if current version is outdated, query new data
#     else:
#         master['ver'] = 2
#         print('updated master')
        
# # if file is empty
# else: 
#     master['ver'] = 1
#     print('write new to master')
print(list(master['data'].keys()))

master.close()