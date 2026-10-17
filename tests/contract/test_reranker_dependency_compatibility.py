"""FlagEmbedding encoder-only reranker 的运行时依赖契约。"""

from __future__ import annotations

from packaging.version import Version
from transformers import XLMRobertaTokenizer, XLMRobertaTokenizerFast, __version__


def test_flagembedding_xlm_roberta_tokenizers_keep_prepare_for_model_api():
    """FlagEmbedding 1.4 的真实打分路径会直接调用该 Transformers 4.x API。"""

    assert Version(__version__) < Version("5.0.0")
    assert callable(getattr(XLMRobertaTokenizer, "prepare_for_model", None))
    assert callable(getattr(XLMRobertaTokenizerFast, "prepare_for_model", None))
