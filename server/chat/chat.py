from fastapi import Body
from fastapi.responses import StreamingResponse
from configs import LLM_MODELS, TEMPERATURE
from server.utils import wrap_done, get_ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable
import asyncio
import json
from langchain.prompts.chat import ChatPromptTemplate
from typing import List, Optional, Union
from server.chat.utils import History
from langchain.prompts import PromptTemplate
from server.utils import get_prompt_template
from server.memory.conversation_db_buffer_memory import ConversationBufferDBMemory
from server.db.repository import add_message_to_db
from server.callback_handler.conversation_callback_handler import ConversationCallbackHandler
from configs import logger


async def chat(
    query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
    conversation_id: str = Body("", description="对话框ID"),
    history_len: int = Body(-1, description="从数据库中取历史消息的数量"),
    history: Union[int, List[History]] = Body(
        [],
        description="历史对话，设为一个整数可以从数据库中读取历史消息",
        examples=[[{
            "role": "user",
            "content": "我们来玩成语接龙，我先来，生龙活虎"
        }, {
            "role": "assistant",
            "content": "虎头虎脑"
        }]]),
    stream: bool = Body(False, description="流式输出"),
    model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
    temperature: float = Body(TEMPERATURE,
                              description="LLM 采样温度",
                              ge=0.0,
                              le=1.0),
    max_tokens: Optional[int] = Body(
        None, description="限制LLM生成Token数量，默认None代表模型最大值"),
    # top_p: float = Body(TOP_P, description="LLM 核采样。勿与temperature同时设置", gt=0.0, lt=1.0),
    prompt_name: str = Body(
        "default", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
):

    async def chat_iterator() -> AsyncIterable[str]:
        nonlocal history, max_tokens
        callback = AsyncIteratorCallbackHandler()
        callbacks = [callback]
        logger.setLevel("INFO")
        memory = None

        if conversation_id:
            message_id = add_message_to_db(chat_type="llm_chat",
                                           query=query,
                                           conversation_id=conversation_id)
            logger.info(f"已将消息添加到数据库。消息ID: {message_id}")
            # 负责保存llm response到message db
            conversation_callback = ConversationCallbackHandler(
                conversation_id=conversation_id,
                message_id=message_id,
                chat_type="llm_chat",
                query=query)
            # 自定义了一个回调函数，并且通过AsyncIteratorCallbackHandler异步处理大模型返回的结果
            callbacks.append(conversation_callback)
            logger.info(f"已为对话ID {conversation_id} 添加对话回调")

        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
        )
        logger.info(f"初始化ChatOpenAI模型。模型名称: {model_name}")
        # 优先使用前端传入的历史消息
        if history:
            history = [History.from_data(h) for h in history]
            logger.info(f"已从输入的 history 数据创建 History 对象列表: {history}")
            prompt_template = get_prompt_template("llm_chat", prompt_name)
            logger.info(f"已获取 llm_chat 的 prompt_template: {prompt_template}")
            input_msg = History(role="user",
                                content=prompt_template).to_msg_template(False)
            logger.info(f"已创建用户输入消息的 History 对象: {input_msg}")
            chat_prompt = ChatPromptTemplate.from_messages(
                [i.to_msg_template() for i in history] + [input_msg])
            logger.info(f"已创建 ChatPromptTemplate 对象: {chat_prompt}")
            logger.info("使用用户提供的历史消息作为聊天提示。")
        elif conversation_id and history_len > 0:  # 前端要求从数据库取历史消息
            # 使用memory 时必须 prompt 必须含有memory.memory_key 对应的变量
            prompt = get_prompt_template("llm_chat", "with_history")
            chat_prompt = PromptTemplate.from_template(prompt)
            logger.info("使用数据库中的历史消息作为聊天提示。")
            # 根据conversation_id 获取message 列表进而拼凑 memory
            memory = ConversationBufferDBMemory(
                conversation_id=conversation_id,
                llm=model,
                message_limit=history_len)

        else:
            prompt_template = get_prompt_template("llm_chat", prompt_name)
            input_msg = History(role="user",
                                content=prompt_template).to_msg_template(False)
            chat_prompt = ChatPromptTemplate.from_messages([input_msg])
            logger.info("使用默认聊天提示，没有历史消息。")
        chain = LLMChain(prompt=chat_prompt, llm=model, memory=memory)
        # Begin a task that runs in the background.
        task = asyncio.create_task(
            wrap_done(
                # acall读取llm数据，没读到一个token，就向AsyncIteratorCallbackHandler中的队列里面写
                chain.acall({"input": query}),
                callback.done), )

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                logger.info(f"xxxstreamxxx==={token}")
                yield json.dumps({
                    "text": token,
                    "message_id": message_id
                },
                                 ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                logger.info(f"xxxxxx==={token}")
                answer += token
            yield json.dumps({
                "text": answer,
                "message_id": message_id
            },
                             ensure_ascii=False)
        # 出发任务执行
        await task

    return StreamingResponse(chat_iterator(), media_type="text/event-stream")
