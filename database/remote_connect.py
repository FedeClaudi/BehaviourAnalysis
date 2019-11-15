# %% 
import datajoint
import pymysql

# %%
conn = pymysql.connect( 
    host = '127.18.0.1', 
    port = 3306, 
    user = 'root', 
    passwd = 'fede', 
    db = 'mysql')
    

# %%
print(conn)

# %%
