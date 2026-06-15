"""Unified LLM interface with automatic language instruction support.

Usage:
    # Simple text generation (auto-appends language instruction)
    answer = LLM.generate("What is Python?")

    # Structured output (auto-appends language instruction)
    result = LLM.generate_structured("Classify: ...", ClassifyOutput)

    # Internal call WITHOUT language instruction (distillation, etc.)
    result = LLM.generate_structured("Distill: ...", DistillOutput, use_language=False)

    # Custom chain pattern
    model = LLM.get_model()
    chain = prompt | model | StrOutputParser()

    # Pre-built chain with auto language instruction
    chain = LLM.build_chain(system_prompt, human_template)
"""

from typing import Optional, Type

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from agent.utils.agent_utils import get_language_instruction, with_retry
from server.config import LLM_MODEL, LLM_TEMPERATURE, LLM_APIKEY


class LLM:
    """Centralized LLM caller. User-facing prompts get language instruction automatically."""

    _default_model: Optional[ChatDeepSeek] = None

    @classmethod
    def get_model(cls, temperature: Optional[float] = None) -> ChatDeepSeek:
        """Get a ChatDeepSeek instance. Shares a default instance when temperature is not overridden."""
        if temperature is not None:
            return ChatDeepSeek(model=LLM_MODEL, temperature=temperature, base_url="https://api.deepseek.com")
        if cls._default_model is None:
            cls._default_model = ChatDeepSeek(model=LLM_MODEL, temperature=LLM_TEMPERATURE, base_url="https://api.deepseek.com")
        return cls._default_model

    @classmethod
    def get_openai_model(cls, base_url: str, model: str, temperature: Optional[float] = 0) -> ChatOpenAI:
        return ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=LLM_APIKEY,
            temperature=temperature
        )


    @classmethod
    def get_specified_model(cls, model: str | None, temperature: Optional[float] = None) -> ChatDeepSeek:
        """Get a ChatDeepSeek instance. Shares a default instance when temperature is not overridden."""
        try:
            if model is not None:
                return ChatDeepSeek(model=model, temperature=temperature)
            return LLM.get_model(temperature)
        except Exception:
            return LLM.get_model(temperature)

    @classmethod
    def get_model_for(cls, task: str, temperature: Optional[float] = None) -> ChatDeepSeek:
        """Get a model instance for a specific task.

        Uses TASK_MODEL_MAP to select the model. Unregistered tasks fall back to LLM_MODEL.
        """
        from server.config import TASK_MODEL_MAP
        model_name = TASK_MODEL_MAP.get(task, LLM_MODEL)
        if temperature is not None:
            return ChatDeepSeek(model=model_name, temperature=temperature)
        if cls._default_model is None:
            cls._default_model = ChatDeepSeek(model=model_name, temperature=LLM_TEMPERATURE)
        return cls._default_model

    @classmethod
    def generate(cls, prompt: str, use_language: bool = True) -> str:
        """Generate text. Language instruction auto-appended unless use_language=False."""
        if use_language:
            prompt += get_language_instruction()
        result = with_retry(lambda: cls.get_model().invoke(prompt))
        return result.content if hasattr(result, "content") else str(result)

    @classmethod
    def generate_structured(
        cls,
        prompt: str,
        output_model: Type[BaseModel],
        model: str = None,
        use_language: bool = True,
    ):
        """Generate structured output. Language instruction auto-appended unless use_language=False."""
        if use_language:
            prompt += get_language_instruction()
        llm_model = cls.get_specified_model(model).with_structured_output(output_model)
        return with_retry(lambda: llm_model.invoke(prompt))

    @classmethod
    def generate_openai_structured(
            cls,
            prompt: str,
            output_model: Type[BaseModel],
            model: str,
            use_language: bool = True,
    ):
        """Generate structured output. Language instruction auto-appended unless use_language=False."""
        if use_language:
            prompt += get_language_instruction()
        llm_model = cls.get_openai_model("https://api.deepseek.com", model).with_structured_output(output_model)
        return with_retry(lambda: llm_model.invoke(prompt))

    @classmethod
    def build_chain(cls, system_prompt: str, human_template: str = "{input}"):
        """Build a system→human chain with language instruction auto-appended to the system prompt.

        Usage:
            chain = LLM.build_chain("You are a helpful assistant.", "{input}")
            result = chain.invoke({"input": "Hello"})
        """
        full_system = system_prompt + get_language_instruction()
        prompt = ChatPromptTemplate.from_messages([
            ("system", full_system),
            ("human", human_template),
        ])
        return prompt | cls.get_model() | StrOutputParser()
