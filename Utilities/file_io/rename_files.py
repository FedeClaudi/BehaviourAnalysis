# %%
import os

fld = r"Z:\branco\Federico\raw_behaviour\maze\analoginputdata"

for f in os.listdir(fld):
    if f[1] == "8":
        corr = list(f)
        corr[1] = "9"
        corr = "".join(corr)
        os.rename(os.path.join(fld, f), os.path.join(fld, corr))
        

#%%
