import pandas as pd
import json
import ast
from datetime import datetime

# --- ОПРЕДЕЛЯЕМ СТРУКТУРУ И ЗАГРУЖАЕМ ДАННЫЕ ПО ЗАЯВКАМ ИЗ ФАЙЛА ---
dtype_spec = {
    'Номер заявки': 'int',            
    'Текст заявки': 'string',  
    'Критичность заявки': 'category',  
    'Категория заявки': 'category',  
    'Длительность работы фактическая, ч.': 'float32'  
            }


#Загружаем список заявок клиента с ГуглДиска
df = pd.read_csv('https://drive.google.com/uc?id=1ilyFdfUA-n0kCDtNwKuwvWMaIOkL4bZ2',
                    sep=';',
                    skipinitialspace=True, 
                    dtype=dtype_spec,
                    decimal=',',
                    parse_dates=['Дата и время создания заявки'],
                    date_format='%d.%m.%Y %H:%M')

#функция для получения трудоемкости заявки по ее номеру
def get_work_for_ticket(ticket_num: str) -> float:

    res_df = df[df['Номер заявки']==int(ticket_num)]
    if res_df.empty:
        return -1.0 #если не можем получить, возвращем -1
    else:
        return float(df[df['Номер заявки']==int(ticket_num)]['Длительность работы фактическая, ч.'].values[0])

#функция для получения списка заявок за период времени
def get_tickets_df(start_str: str,end_str: str) -> pd.DataFrame:
    
    #нормализуем полученные строки с датами
    start_str = normalize_date_string(start_str)
    end_str = normalize_date_string(end_str)

    #если получена пустая строка - возвращаем пустой DataFrame
    if start_str.strip() == '' or end_str.strip() == '':
        return pd.DataFrame()
    
    #конвертируем строку в DateTime
    start_date = pd.to_datetime(start_str,format='%d.%m.%Y %H:%M')
    end_date = pd.to_datetime(end_str,format='%d.%m.%Y %H:%M')
    
    #производим фильтрацию DataFrame
    return df[df['Дата и время создания заявки'].between(start_date, end_date, inclusive='both')]

# функция для нормализации строки с датой
def normalize_date_string(date_str, target_format='%d.%m.%Y %H:%M'):
    """
    Преобразует любую строку с датой в целевой формат
    
    Parameters:
    -----------
    date_str : str
        Входная строка с датой в любом формате
    target_format : str
        Целевой формат (по умолчанию '%d.%m.%Y %H:%M')
    
    Returns:
    --------
    str
        Дата в целевом формате или пустая строка при ошибке
    """
    if pd.isna(date_str) or str(date_str).strip() == '':
        return ''
    
    try:
        # Преобразуем в datetime (pandas сам определит большинство форматов)
        dt = pd.to_datetime(date_str, dayfirst=False, errors='coerce')
        
        if pd.isna(dt):
            # Если pandas не справился            
            return ''
        
        # Преобразуем в целевой формат
        return dt.strftime(target_format)
    
    except Exception as e:        
        return ''

# Блок для быстрого теста
if __name__ == "__main__":
    
    print(df.dtypes)
    

