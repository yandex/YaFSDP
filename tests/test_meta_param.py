import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from ya_fsdp.meta_param import convert_some_params_to_metaparams  # noqa: E402


class TiedEmbeddingHead(nn.Module):
    """Model with a tied weight between the embedding and the projection head.

    This mirrors the weight tying used by decoder-only LLMs (e.g. LLaMA),
    where ``lm_head.weight`` is the same ``nn.Parameter`` as
    ``embed_tokens.weight``.
    """

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        # Tie the weights: both submodules now share a single nn.Parameter.
        self.head.weight = self.embed.weight


def test_shared_parameter_is_converted_once():
    model = TiedEmbeddingHead(vocab_size=16, hidden_size=8)

    metaparams = convert_some_params_to_metaparams(model, "", lambda _: True)

    # There are two `nn.Parameter` attributes (embed.weight and head.weight)
    # but they reference the *same* tensor, so it must be converted exactly
    # once to avoid double-counting its elements in the super-tensor buffer.
    assert len(metaparams) == 1
    assert metaparams[0].numel == 16 * 8
    assert metaparams[0].param is model.embed.weight


def test_distinct_parameters_are_all_converted():
    class TwoParams(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.randn(4))
            self.b = nn.Parameter(torch.randn(5))

    model = TwoParams()
    metaparams = convert_some_params_to_metaparams(model, "", lambda _: True)
    assert len(metaparams) == 2
    assert sum(p.numel for p in metaparams) == 9
