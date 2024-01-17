from fastapi import Body
from configs import logger, log_verbose, LLM_MODELS, HTTPX_DEFAULT_TIMEOUT
from server.utils import (BaseResponse, fschat_controller_address, list_config_llm_models,
                          get_httpx_client, get_model_worker_config)
from typing import List


def list_running_models(
    controller_address: str = Body(None, description="Fastchat controller服务器地址", examples=[fschat_controller_address()]),
    placeholder: str = Body(None, description="该参数未使用，占位用"),
) -> BaseResponse:
    '''
    从fastchat controller获取已加载模型列表及其配置项
    '''
    try:
        controller_address = controller_address or fschat_controller_address()
        # 在 Python 的 get_httpx_client() 上下文中，with 语句用于确保资源（如网络连接）在使用后被适当地释放或关闭。具体来说，当你在使用 with 语句时，Python 会自动调用一些特殊方法，例如 __enter__() 和 __exit__()，来处理资源的创建和销毁。
        # 以 get_httpx_client() 为例，这个函数可能返回一个 HTTP 客户端对象，该对象用于发送 HTTP 请求。使用 with 语句可以确保在代码块执行完毕后，这个客户端对象被适当地关闭或释放，从而避免资源泄漏。
        with get_httpx_client() as client:
            r = client.post(controller_address + "/list_models")
            # 这是一个 Python 代码片段，它从响应中提取 JSON 数据，并从中获取 "models" 字段的值。
            # r 是一个响应对象，它通常是由 HTTP 客户端（如 requests 或 httpx）返回的。r.json() 方法用于将响应内容解析为 JSON 对象。
            # ["models"] 是一个索引表达式，用于从 JSON 对象中获取 "models" 字段的值。这个字段通常是一个数组，包含了模型对象的信息。
            models = r.json()["models"]
            data = {m: get_model_config(m).data for m in models}
            return BaseResponse(data=data)
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}',
                        exc_info=e if log_verbose else None)
        return BaseResponse(
            code=500,
            data={},
            msg=f"failed to get available models from controller: {controller_address}。错误信息是： {e}")


def list_config_models(
    types: List[str] = Body(["local", "online"], description="模型配置项类别，如local, online, worker"),
    placeholder: str = Body(None, description="占位用，无实际效果")
) -> BaseResponse:
    '''
    从本地获取configs中配置的模型列表
    '''
    data = {}
    for type, models in list_config_llm_models().items():
        if type in types:
            data[type] = {m: get_model_config(m).data for m in models}
    return BaseResponse(data=data)


def get_model_config(
    model_name: str = Body(description="配置中LLM模型的名称"),
    placeholder: str = Body(None, description="占位用，无实际效果")
) -> BaseResponse:
    '''
    获取LLM模型配置项（合并后的）
    '''
    config = {}
    # 删除ONLINE_MODEL配置中的敏感信息
    for k, v in get_model_worker_config(model_name=model_name).items():
        if not (k == "worker_class"
            or "key" in k.lower()
            or "secret" in k.lower()
            or k.lower().endswith("id")):
            config[k] = v

    return BaseResponse(data=config)


def stop_llm_model(
    model_name: str = Body(..., description="要停止的LLM模型名称", examples=[LLM_MODELS[0]]),
    controller_address: str = Body(None, description="Fastchat controller服务器地址", examples=[fschat_controller_address()])
) -> BaseResponse:
    '''
    向fastchat controller请求停止某个LLM模型。
    注意：由于Fastchat的实现方式，实际上是把LLM模型所在的model_worker停掉。
    '''
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(
                controller_address + "/release_worker",
                json={"model_name": model_name},
            )
            return r.json()
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}',
                        exc_info=e if log_verbose else None)
        return BaseResponse(
            code=500,
            msg=f"failed to stop LLM model {model_name} from controller: {controller_address}。错误信息是： {e}")


def change_llm_model(
    model_name: str = Body(..., description="当前运行模型", examples=[LLM_MODELS[0]]),
    new_model_name: str = Body(..., description="要切换的新模型", examples=[LLM_MODELS[0]]),
    controller_address: str = Body(None, description="Fastchat controller服务器地址", examples=[fschat_controller_address()])
):
    '''
    向fastchat controller请求切换LLM模型。
    '''
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(
                controller_address + "/release_worker",
                json={"model_name": model_name, "new_model_name": new_model_name},
                timeout=HTTPX_DEFAULT_TIMEOUT, # wait for new worker_model
            )
            return r.json()
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}',
                        exc_info=e if log_verbose else None)
        return BaseResponse(
            code=500,
            msg=f"failed to switch LLM model from controller: {controller_address}。错误信息是： {e}")


def list_search_engines() -> BaseResponse:
    from server.chat.search_engine_chat import SEARCH_ENGINES

    return BaseResponse(data=list(SEARCH_ENGINES))
