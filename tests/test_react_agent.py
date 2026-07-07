import sys
import types
import unittest


def install_langchain_stubs():
    """Provide the tiny LangChain surface ReActAgent needs for these tests."""
    ollama_module = types.ModuleType("langchain_ollama")

    class OllamaLLM:
        def __init__(self, *args, **kwargs):
            pass

    ollama_module.OllamaLLM = OllamaLLM
    sys.modules.setdefault("langchain_ollama", ollama_module)

    prompts_module = types.ModuleType("langchain_core.prompts")

    class PromptTemplate:
        def __init__(self, template, input_variables):
            self.template = template
            self.input_variables = input_variables

        def format(self, **kwargs):
            return self.template.format(**kwargs)

    prompts_module.PromptTemplate = PromptTemplate
    sys.modules.setdefault("langchain_core.prompts", prompts_module)

    messages_module = types.ModuleType("langchain_core.messages")

    class BaseMessage:
        pass

    class HumanMessage:
        pass

    class AIMessage:
        pass

    messages_module.BaseMessage = BaseMessage
    messages_module.HumanMessage = HumanMessage
    messages_module.AIMessage = AIMessage
    sys.modules.setdefault("langchain_core.messages", messages_module)


install_langchain_stubs()

from src.react_agent import ReActAgent


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def invoke(self, prompt):
        if not self.responses:
            raise AssertionError("No fake LLM responses left")
        return self.responses.pop(0)


class ReActAgentTraceTests(unittest.TestCase):
    def test_direct_final_answer_is_recorded_as_trace_step(self):
        agent = ReActAgent(
            llm=FakeLLM(["Thought: I can answer this directly.\nFinal Answer: 15% of 200 is 30."]),
            tools={},
            verbose=False,
        )

        answer, trace = agent.think("Calculate 15% of 200")

        self.assertEqual(answer, "15% of 200 is 30.")
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0].action, "final_answer")
        self.assertEqual(trace[0].action_input, "15% of 200 is 30.")

    def test_final_answer_after_tool_keeps_tool_trace(self):
        agent = ReActAgent(
            llm=FakeLLM([
                "Thought: I should calculate it.\nAction: calculate\nAction Input: 200 * 0.15",
                "Thought: I now know the final answer.\nFinal Answer: 30",
            ]),
            tools={"calculate": lambda expression: 30},
            verbose=False,
        )

        answer, trace = agent.think("Calculate 15% of 200")

        self.assertEqual(answer, "30")
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0].action, "calculate")
        self.assertEqual(trace[0].observation, "30")


if __name__ == "__main__":
    unittest.main()
