from typing import TYPE_CHECKING
import json
import time

from dotenv import load_dotenv

import pandas as pd

# Импорты LangChain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langfuse.langchain import CallbackHandler

# Импорты LangGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# TICKETS_ENGINE
from tickets_engine import get_tickets_df,get_work_for_ticket



if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.runnables import RunnableConfig

# --- НАСТРОЙКА ---
load_dotenv()

@tool
def get_tickets(begDate: str,endDate: str) -> str:
    """
    Используй этот инструмент, чтобы получить список заявок клиентов за период времени. 
    Args:
        begDate: Дата начала периода
        endDate: Дата окончания периода
    
    Returns:
        JSON массив с информацией о заявках (ID - ID заявки, DATE_CREATE - дата создания заявки, TEXT - текст заявки, CATEGORY - категория заявки):
        [{"ID":1,"DATE_CREATE":"2025-01-29 16:47:00","TEXT":"Текст заявки","CATEGORY":"Инцидент"},
        {"ID":2,"DATE_CREATE":"2025-01-25 16:47:00,"TEXT":"Текст заявки 2","CATEGORY":"Настройка"},
        {"ID":3,"DATE_CREATE":"2025-01-30 16:47:00,"TEXT":"Текст заявки 3","CATEGORY":"Консультация"}]            
        
    """
    start_time = time.time()
    print(f"-- Получение списка заявок...",end='\n')
    df = get_tickets_df(begDate,endDate)

    columns_to_keep = ['Номер заявки','Дата и время создания заявки','Текст заявки','Категория заявки']
    df_filtered = df[columns_to_keep]

    rename_dict = {
    'Номер заявки': 'ID',
    'Дата и время создания заявки': 'DATE_CREATE',
    'Текст заявки': 'TEXT',
    'Категория заявки': 'CATEGORY'
    }

    df_renamed = df_filtered.rename(columns=rename_dict)
    res = df_renamed.to_json(force_ascii=False,orient='records', lines=True,date_format='iso')    
    end_time = time.time()
    print(f"-- Время получения списка заявок: {end_time - start_time:.4f} секунд",end='\n')

    return res
    

@tool
def get_work(ticket_id:int) -> float:
    """
    Используй этот инструмент, чтобы получить  трудоемкость заявки по ее ID.
    Args:
        ticket_id: ID заявки
    Returns:
        Трудоемкость заявки, float
    """
    start_time = time.time()
    print(f"-- Получение трудоемкости заявки {ticket_id}...",end='\n')
    
    res = get_work_for_ticket(ticket_id)
            
    end_time = time.time()    
    print(f"-- Время получения трудоемкости заявки: {end_time - start_time:.4f} секунд",end='\n')

    return res


tools = [get_tickets,get_work]

#2. Модель
llm = ChatOllama(
    #model="qwen3:8b",
    model="deepseek-v3.1:671b-cloud",    
    base_url="http://localhost:11434",
    temperature=0,
)
llm_with_tools = llm.bind_tools(tools)

# 3. Память
memory = MemorySaver()


# --- ГРАФ ---


def call_model(state: MessagesState):
    """Узел агента"""

    start_time = time.time()
    print(f"-- Выполнение запроса к модели...",end='\n')

    messages = state["messages"]
    response = llm_with_tools.invoke(messages)

    end_time = time.time()
    print(f"-- Время выполнения запроса к модели: {end_time - start_time:.4f} секунд",end='\n')

    return {"messages": [response]}


workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

app = workflow.compile(checkpointer=memory)


# --- ИНТЕРФЕЙС ---
def main():

    print("🤖 Ассистент анализа заявок клиентов готов к работе! (Введите 'q' для выхода)")

    # 1. Инициализируем хендлер
    try:
        langfuse_handler = CallbackHandler()
        print("✅ Langfuse мониторинг подключен")
    except Exception as e:
        print(f"⚠️ Ошибка подключения Langfuse: {e}")
        langfuse_handler = None

    # 2. Добавляем его в конфиг
    callbacks: list[BaseCallbackHandler] = [langfuse_handler] if langfuse_handler else []

    config: RunnableConfig = {
        "configurable": {"thread_id": "session_1"},
        "callbacks": callbacks,
    }

    sys_msg = SystemMessage(
        content=
        """Ты ассистент с доступом к инструментам (tools).
            Когда пользователь просит что-то, что требует внешних данных или вычислений, используй доступные инструменты.
            
            Доступные инструменты:
            1. get_tickets(begDate, endDate) - получение списка заявок за указанный период.
            2. get_work(ticket_ID) - получение трудоемкости заявки по ID заявки.  """        
    )

    while True:
        try:
            user_input = input("\nВы: ")
            if user_input.lower() in ["q", "exit", "quit"]:
                print("До свидания!")
                break

            inputs: MessagesState = {
                "messages": [sys_msg, HumanMessage(content=user_input)],
            }            

            for event in app.stream(inputs, config=config):
                if "agent" in event:
                    print(".", end='\n', flush=True)
                if "tools" in event:
                    print(" [Испольование инструмента] ", end='\n', flush=True)

            snapshot = app.get_state(config)
            if snapshot.values["messages"]:
                last_message = snapshot.values["messages"][-1]
                if hasattr(last_message, "content"):
                    print(f"\n\n🤖 Ассистент:\n{last_message.content}")

        except KeyboardInterrupt:
            print("\nВыход...")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")


if __name__ == "__main__":
    main()