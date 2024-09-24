import glob 
import numpy as np
import pandas as pd


glob_list_ct = glob.glob("/mnt/d/Github/pytorch-stable-diffusion/immagini_zippate/*/*ct*")
print(glob_list_ct)
glob_list_t1w = [path.replace("ct","T1w") for path in glob_list_ct]

prompt_list = ["transform this t1w to ct scan" for x in range(len(glob_list_ct))]

dict_for_pandas = {
    "input_image": glob_list_t1w,
    "edited_image": glob_list_ct,
    "edit_prompt": prompt_list
}
df = pd.DataFrame(data=dict_for_pandas)
df.to_csv("csv_t1w_to_ct.csv", index=False)

prompt_list = ["transform this ct scan to t1w" for x in range(len(glob_list_ct))]

dict_for_pandas = {
    "input_image": glob_list_ct,
    "edited_image": glob_list_t1w,
    "edit_prompt": prompt_list
}
df = pd.DataFrame(data=dict_for_pandas)
df.to_csv("csv_ct_to_t1w.csv", index=False)
