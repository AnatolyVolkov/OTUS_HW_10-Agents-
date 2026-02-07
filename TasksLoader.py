import pandas as pd

def loadTickets(fileLink:str)->pd.DataFrame:

  df = pd.read_csv(fileLink,
                 sep=';',
                 encoding='utf-8',
                 on_bad_lines='skip',
                engine='pyarrow')
  return df;