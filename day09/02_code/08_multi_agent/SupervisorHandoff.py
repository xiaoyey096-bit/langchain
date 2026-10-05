import os
from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START
from langgraph.graph.message import MessagesState
from langgraph.prebuilt.tool_node import InjectedState
from langgraph.types import Command, Send


# ===============================
# 1. 初始化大语言模型
# ===============================
def init_llm_model() -> ChatOpenAI:
    return ChatOpenAI(
        model="qwen-plus",
        api_key=os.getenv("aliQwen-api"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.1,
        max_tokens=1024,
    )


model = init_llm_model()


# ===============================
# 2. 通用 Handoff 工具工厂
# ===============================
def create_task_description_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"移交给 {agent_name}"

    @tool(name, description=description)
    def handoff_tool(
        task_description: Annotated[str,"描述下一个 Agent 应该做什么，包括所有必要信息"],
        state: Annotated[MessagesState, InjectedState],
    ) -> Command:
        task_description_message = {
            "role": "user",
            "content": task_description,
        }
        agent_input = {
            **state,
            "messages": [task_description_message],
        }

        return Command(
            goto=[Send(agent_name, agent_input)],
            graph=Command.PARENT,
        )

    return handoff_tool


# ===============================
# 3. 业务工具（必须有 docstring）
# ===============================
@tool("book_flight")
def book_flight(from_airport: str, to_airport: str) -> str:
    """预订航班，根据出发地和目的地完成机票预订"""
    print(f"✅ 成功预订了从 {from_airport} 到 {to_airport} 的航班")
    return f"成功预订了从 {from_airport} 到 {to_airport} 的航班。"


@tool("book_hotel")
def book_hotel(hotel_name: str) -> str:
    """预订酒店，根据酒店名称完成预订"""
    print(f"✅ 成功预订了 {hotel_name} 的住宿")
    return f"成功预订了 {hotel_name} 的住宿。"


# ===============================
# 4. Handoff 工具
# ===============================
transfer_to_flight_assistant = create_task_description_handoff_tool(
    agent_name="flight_assistant",
    description="将任务移交给航班预订助手",
)

transfer_to_hotel_assistant = create_task_description_handoff_tool(
    agent_name="hotel_assistant",
    description="将任务移交给酒店预订助手",
)


# ===============================
# 5. 定义 Agent（create_agent 新接口）
# create_agent 不再显式接收 prompt，而是：
# 通过 tool schema + tool 名称 + tool docstring
# 通过 graph 上下文（handoff 描述）
# 通过 MessagesState 历史消息
# ===============================
flight_assistant = create_agent(
    model=model,
    tools=[book_flight, transfer_to_hotel_assistant], # 包含移交工具
    name="flight_assistant",
)

hotel_assistant = create_agent(
    model=model,
    tools=[book_hotel, transfer_to_flight_assistant], # 包含移交工具
    name="hotel_assistant",
)


# ===============================
# 6. 构建多 Agent Graph
# ===============================
multi_agent_graph = (
    StateGraph(MessagesState)
    .add_node(flight_assistant)
    .add_node(hotel_assistant)
    .add_edge(START, "flight_assistant")
    .compile()
)


# ===============================
# 7. 运行
# ===============================
if __name__ == "__main__":
    result = multi_agent_graph.invoke(
        {
            "messages": [
                HumanMessage(content="帮我预订从北京到上海的航班，并预订如家酒店")
            ]
        }
    )

    print("\n====== 最终对话结果 ======")
    for msg in result["messages"]:
        if msg.type in ("human", "ai"):
            print(msg.content)

# 外层主图：START
#     ↓ 静态边
# 外层主图：进入 flight_assistant【子图开始运行】
#     ↓ 子图内部ReAct循环
# LLM：需要两件事，订航班 + 订酒店。当前Agent只能订航班，酒店交给hotel_assistant
# LLM选择调用工具 transfer_to_hotel_assistant（handoff_tool）
#     ↓ 执行 handoff_tool
# 构造 agent_input，return Command(goto=[Send("hotel_assistant", agent_input)], graph=Command.PARENT)
#     ← graph=PARENT，跳出flight子图，回到外层主图，执行动态跳转
# 外层主图：切换到 hotel_assistant【子图开始运行】
#     ↓ hotel子图内部ReAct循环
# LLM调用 book_hotel 工具，预订如家酒店
# 工具返回成功结果，消息写入messages
# LLM再次思考：所有任务全部做完，**不再调用任何工具，不返回Command**
# hotel_assistant子图执行完毕，退出，交还控制权给外层主图
# 外层主图：没有任何下一跳节点，没有Command跳转
# 👉 图自动结束，到达END

