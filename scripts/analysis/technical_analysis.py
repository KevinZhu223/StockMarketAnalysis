import pandas as pd
import numpy as np
from scripts.analysis.feature_engineering import add_tech_indicators

def calc_advanced_indicators(df):
    df = add_tech_indicators(df)
    
    #Ichimoku Cloud