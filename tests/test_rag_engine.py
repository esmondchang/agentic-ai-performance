import sys
import types
import unittest


def install_rag_stubs():
    ollama_module = types.ModuleType("langchain_ollama")

    class OllamaLLM:
        def __init__(self, model=None, temperature=None):
            self.model = model
            self.temperature = temperature

        def invoke(self, prompt):
            return "stubbed response"

    class OllamaEmbeddings:
        def __init__(self, model=None):
            self.model = model

    ollama_module.OllamaLLM = OllamaLLM
    ollama_module.OllamaEmbeddings = OllamaEmbeddings
    sys.modules.setdefault("langchain_ollama", ollama_module)

    vectorstores_module = types.ModuleType("langchain_community.vectorstores")

    class FakeFAISS:
        def __init__(self):
            self.docs = []

        @classmethod
        def from_texts(cls, texts, embeddings, metadatas=None):
            store = cls()
            store.docs = [
                Document(page_content=text, metadata=(metadatas or [{}] * len(texts))[i])
                for i, text in enumerate(texts)
            ]
            return store

        @classmethod
        def load_local(cls, index_path, embeddings, allow_dangerous_deserialization=False):
            return cls()

        def add_documents(self, docs):
            self.docs.extend(docs)

        def save_local(self, index_path):
            pass

        def similarity_search(self, query, k=5, filter=None):
            return self.docs[:k]

        def similarity_search_with_relevance_scores(self, query, k=5):
            return [(doc, 0.9) for doc in self.docs[:k] if hasattr(doc, "page_content")]

    vectorstores_module.FAISS = FakeFAISS
    sys.modules.setdefault("langchain_community.vectorstores", vectorstores_module)

    splitters_module = types.ModuleType("langchain_text_splitters")

    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size, chunk_overlap, length_function, separators):
            self._chunk_size = chunk_size
            self._chunk_overlap = chunk_overlap

        def split_text(self, text):
            return [text.strip()]

    splitters_module.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
    sys.modules.setdefault("langchain_text_splitters", splitters_module)

    documents_module = types.ModuleType("langchain_core.documents")

    class Document:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

    documents_module.Document = Document
    sys.modules.setdefault("langchain_core.documents", documents_module)

    prompts_module = types.ModuleType("langchain_core.prompts")

    class PromptTemplate:
        def __init__(self, template, input_variables):
            self.template = template
            self.input_variables = input_variables

        def format(self, **kwargs):
            return self.template.format(**kwargs)

    prompts_module.PromptTemplate = PromptTemplate
    sys.modules.setdefault("langchain_core.prompts", prompts_module)
    sys.modules.setdefault("faiss", types.ModuleType("faiss"))


install_rag_stubs()

from src.rag_engine import RAGEngine


class RAGEngineLiveDocumentTests(unittest.TestCase):
    def test_fetch_latest_financial_documents_uses_yfinance_data(self):
        class FakeFastInfo(dict):
            last_price = 45.25
            previous_close = 44.5
            last_volume = 123456

        class FakeTicker:
            info = {
                "longName": "Super Micro Computer, Inc.",
                "sector": "Technology",
                "industry": "Computer Hardware",
                "marketCap": 1000000000,
                "trailingPE": 13.9,
                "forwardPE": 8.3,
                "trailingEps": 3.25,
                "targetMeanPrice": 52.0,
            }
            fast_info = FakeFastInfo()
            news = [
                {
                    "title": "SMCI announces updated outlook",
                    "publisher": "Example Finance",
                    "link": "https://example.com/smci",
                    "providerPublishTime": 1783440000,
                    "summary": "Current news summary.",
                }
            ]

            def __init__(self, ticker):
                self.ticker = ticker

        fake_yfinance = types.ModuleType("yfinance")
        fake_yfinance.Ticker = FakeTicker
        previous_yfinance = sys.modules.get("yfinance")
        sys.modules["yfinance"] = fake_yfinance

        try:
            rag = RAGEngine()
            docs = rag.fetch_latest_financial_documents("smci")
        finally:
            if previous_yfinance is None:
                sys.modules.pop("yfinance", None)
            else:
                sys.modules["yfinance"] = previous_yfinance

        self.assertGreaterEqual(len(docs), 3)
        self.assertTrue(all(doc["metadata"]["is_live"] for doc in docs))
        self.assertIn("45.25", docs[0]["content"])
        self.assertIn("Trailing P/E: 13.9", docs[1]["content"])
        self.assertEqual(docs[2]["metadata"]["document_type"], "news")

    def test_sample_documents_are_clearly_labeled_not_live(self):
        rag = RAGEngine()

        docs = rag._get_sample_financial_documents("AAPL")

        self.assertTrue(all(doc["metadata"]["is_live"] is False for doc in docs))
        self.assertTrue(all("SAMPLE DATA ONLY" in doc["content"] for doc in docs))

    def test_add_documents_registers_new_source_type(self):
        rag = RAGEngine()

        count = rag.add_documents(
            [{"content": "Latest live document", "metadata": {"ticker": "SMCI"}}],
            "live_financial_data",
        )

        self.assertEqual(count, 1)
        self.assertIn("live_financial_data", rag.document_sources)
        self.assertEqual(len(rag.document_sources["live_financial_data"]), 1)

    def test_load_financial_documents_replaces_old_index_with_live_docs(self):
        rag = RAGEngine()
        rag.add_documents(
            [{"content": "SAMPLE DATA ONLY - old 2024 data", "metadata": {"is_live": False}}],
            "sample_financial_reports",
        )
        rag.fetch_latest_financial_documents = lambda ticker: [
            {
                "content": "Latest market snapshot for SMCI with current live data",
                "metadata": {"ticker": "SMCI", "is_live": True},
            }
        ]

        count = rag.load_financial_documents("SMCI")
        docs = rag.retrieve("2024", k=5)

        self.assertEqual(count, 1)
        self.assertEqual(list(rag.document_sources.keys()), ["live_financial_data"])
        self.assertEqual(len(docs), 1)
        self.assertIn("Latest market snapshot", docs[0].page_content)
        self.assertNotIn("2024", docs[0].page_content)


if __name__ == "__main__":
    unittest.main()
