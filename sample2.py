#%%
import numpy as np
# %%
!pip list
# %%
!nvidia-smi
# %%
!conda activate pytorch
# %%
conda which
# %%
sins = np.sin([i*0.05 for i in range(100)])
randoms = np.random.rand(100)
xs = np.array([i*0.05 for i in range(100)]) 
import plotly.graph_objects as go
fig = go.Figure(data=[
    go.Scatter(x=xs, y=sins, name="sin"),
    go.Scatter(x=xs, y=randoms, name="random"),
])
fig.show()
# %%
