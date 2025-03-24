import streamlit as st
import pandas as pd
import numpy as np

st.title("Dashboard de Ejemplo")
data = pd.DataFrame(np.random.randn(10, 3), columns=["A", "B", "C"])
st.line_chart(data)
    