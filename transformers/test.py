from networks import Transformers


model = Transformers(
    encoder_depth=6,
    decoder_depth=6,
    enc_heads=6,
    dec_heads=6,
    d_q=64,
    d_k=64,
)

src = torch.tensor([[
    model.word2idx["Hello"],
    model.word2idx["world"],
    model.word2idx["!"]
]])

tgt = torch.tensor([[
    model.word2idx["Bonjour"],
    model.word2idx["le"],
    model.word2idx["monde"],
    model.word2idx["!"]
]])

output = model(src, tgt)

translated = model.generate(src)

print(translated)
