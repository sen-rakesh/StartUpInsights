#adding necessary libraries
import streamlit as st

import pandas as pd
import numpy as np
np.random.seed(42)
import altair as alt
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu


st.set_page_config(page_title = "Home")
st.markdown(""" # StartUpInsight 

#### by Three Degrees 
for Bitathon'23 """)

st.image("Startups.webp", width = 650)


st.markdown(""" 
## Introduction  
###### For many new businesses, securing funding is crucial for growth and success, but it can be challenging to identify and connect with investors who are interested in their specific industry or business idea. This problem is particularly acute for start-ups that are operating in niche or emerging markets, where traditional funding sources may be limited.
Additionally, start-ups may struggle to create effective pitches or navigate the fundraising process, which can further hinder their ability to secure funding.


###### The real-time problem that “Three Degrees” aims to solve is the difficulty that start-ups face in finding potential investors. "Three Degrees” addresses these challenges by providing a platform that connects start-ups with a network of investors who are interested in funding new ventures.\n 
_1. This unified app platform offers a user-friendly interface that simplifies the process and connecting with investors._\n 
_2. Start-ups can search for investors based on their funding preferences, industry focus, and geographical location, which can help to streamline the process and improve their chances of finding the right investor._\n
###### By addressing these real-time challenges, “Three Degrees” is helping to level the playing field for start-ups and improve their access to funding, which can ultimately contribute to a more vibrant and dynamic start-up ecosystem.\n 
""")




