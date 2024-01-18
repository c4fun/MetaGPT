#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : zhipuai LLM from https://open.bigmodel.cn/dev/api#sdk

import json
from enum import Enum

import openai
from zhipuai import ZhipuAI
from requests import ConnectionError
from tenacity import (
    after_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from metagpt.config import CONFIG, LLMProviderEnum
from metagpt.logs import log_llm_stream, logger
from metagpt.provider.base_llm import BaseLLM
from metagpt.provider.llm_provider_registry import register_provider
from metagpt.provider.openai_api import log_and_reraise
# from metagpt.provider.zhipuai.zhipu_model_api import ZhiPuModelAPI

from zhipuai.types.chat.chat_completion import Completion, CompletionChoice, CompletionMessage
from zhipuai.types.chat.chat_completion import CompletionUsage
from zhipuai.types.chat.chat_completion_chunk import ChoiceDelta
from zhipuai.types.chat.chat_completion_chunk import CompletionUsage as ChunkCompletionUsage

class ZhiPuEvent(Enum):
    ADD = "add"
    ERROR = "error"
    INTERRUPTED = "interrupted"
    FINISH = "finish"


@register_provider(LLMProviderEnum.ZHIPUAI)
class ZhiPuAILLM(BaseLLM):
    """
    Refs to `https://open.bigmodel.cn/dev/api#chatglm_turbo`
    From now, there is only one model named `chatglm_turbo`
    """

    def __init__(self):
        self.llmclient = ZhipuAI(api_key=CONFIG.zhipuai_api_key)
        self.model = "glm-4"  # so far only one model, just use it
        self.use_system_prompt: bool = False  # zhipuai has no system prompt when use api


    def _const_kwargs(self, messages: list[dict]) -> dict:
        kwargs = {"model": self.model, "messages": messages, "temperature": 0.3}
        return kwargs

    def _update_costs(self, usage):
        """update each request's token cost"""
        if CONFIG.calc_usage:
            try:
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
                CONFIG.cost_manager.update_cost(prompt_tokens, completion_tokens, self.model)
            except Exception as e:
                logger.error(f"zhipuai updates costs failed! exp: {e}")

    def get_choice_text(self, resp: dict) -> str:
        """get the first text of choice from llm response"""
        # assist_msg = resp.get("choices", [{"role": "error"}])[-1]
        # assert assist_msg["role"] == "assistant"
        # return assist_msg.get("content")

        usage = resp.usage
        assert isinstance(usage, CompletionUsage)
        self._update_costs(usage)

        assert isinstance(resp, Completion) # assert the type is a completion
        assist_choice = resp.choices[-1]
        assert isinstance(assist_choice, CompletionChoice)
        assist_msg = assist_choice.message
        assert isinstance(assist_msg, CompletionMessage)
        return assist_msg.content

        # assist_msg = resp
        # print(resp)
        # return assist_msg.get("content")

    def completion(self, messages: list[dict], timeout=3) -> dict:
        # resp = self.llm(**self._const_kwargs(messages))
        resp = self.llmclient.chat.completions.create(**self._const_kwargs(messages))
        print(f'Type of resp in completion: {type(resp)}')
        # usage = resp.get("data").get("usage")
        # self._update_costs(usage)
        return resp

    async def _achat_completion(self, messages: list[dict], timeout=3) -> dict:
        resp = self.llmclient.chat.completions.create(**self._const_kwargs(messages))
        # print(f'Type of resp in _achat_completion: {type(resp)}')
        usage = resp.usage
        self._update_costs(usage)
        return resp

    async def acompletion(self, messages: list[dict], timeout=3) -> dict:
        return await self._achat_completion(messages, timeout=timeout)

    async def _achat_completion_stream(self, messages: list[dict], timeout=3) -> str:
        # response = self.llmclient.chat.asyncCompletions.create(**self._const_kwargs(messages))
        response = self.llmclient.chat.completions.create(**self._const_kwargs(messages),stream=True)
        collected_content = []
        usage = {}
        for chunk in response:

            # TODO：解决usage的问题
            # # 不知道为啥usage是none,改到下面也没有好转。可能跟定义中的可选有关
            # print(f'Type of chunk: {type(chunk)}')  # Type of chunk: <class 'zhipuai.types.chat.chat_completion_chunk.ChatCompletionChunk'>
            # usage = chunk.usage
            # print(f'Type of usage: {type(usage)}')  # Type of usage: <class 'NoneType'>
            # # self._update_costs(usage)

            choice_delta = chunk.choices[0].delta
            assert isinstance(choice_delta, ChoiceDelta)
            content = choice_delta.content
            collected_content.append(content)
            log_llm_stream(content)
        log_llm_stream("\n")

        full_content = "".join(collected_content)
        return full_content


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=60),
        after=after_log(logger, logger.level("WARNING").name),
        retry=retry_if_exception_type(ConnectionError),
        retry_error_callback=log_and_reraise,
    )
    async def acompletion_text(self, messages: list[dict], stream=False, timeout=3) -> str:
        """response in async with stream or non-stream mode"""
        if stream:
            return await self._achat_completion_stream(messages)
        resp = await self._achat_completion(messages)
        return self.get_choice_text(resp)
