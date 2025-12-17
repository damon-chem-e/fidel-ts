Their proposition is basically saying "hey look, if you don't condition on external information (external to historical time series), you get the conditional expectation only conditional on the past time series" which is a tautology in a sense. Basically we need a conditional expectation, but we just need to condition on more things (their 'external stimuli' is just more conditioning information into a conditional expectation).

FIATS is LLM-free, numerical based. Would be great to add into an ensemble.

I think a lot of the FIATS gains are from channel specificity. Small datasets and studies used shared parameters per channel. If you have enough data, learning channel-specific parameters is essential, which FIATS captures. They have channel specificity in boht their text head and in their time series head (once linked with text). They don't really have channel specificity in the ts backbone.

The CASM blocks essentially use KV-Q attention between a channel description and the news itself. Interesting! Again, this requires enough data to learn everything (low data regimes is where LLMs come in handy). This is something I can adapt into my models in the text side. Perhaps alternate this attention block (citing) with self-attention on the news itself. This could be incredibly useful for product description and firm description (channel descriptions), though since I prepend that to news, it's essentially subsumed in my methodology (but mention, and try doing this if my stuff doesn't seem to work very well).

Unfolds the non-overlapping patches into overlapping patches. I think this is cool. Wonder if PatchTST does this.

FIATS also uses full bidirectional attention.

FIATS uses a filtering mechanism to find the samples in which text-guided is useful. That's the point of the gate. They didn't put it in their architecture, but they had the concept there and used it to filter the training samples (seems bad).

---

a chat on fiats:

Wait... does TGTSF not have any global text representations that all time series attend to? Meaning the cross attention is specifically only between channel specific time series and channel specific representations? 

Reviewing TGTSF's architecture to clarify how text and time series interact:


[4 tools called]


Yes. In TGTSF, the cross-attention is channel-specific, not global.

Looking at `text_temp_cross_block` (the mixer):

```82:115:layers/TGTSF_torch.py
    def forward(self, text_emb, temp_emb):
        # text_emb:         [b, l, c, d]    Embed the news according to the description of each timestamp and each channel
        # temp_emb:         [b, n, c, d]    embedding of temporal transformers

        B, L_text, C, D_text = text_emb.shape
        _, L_temp, _, D_temp = temp_emb.shape

        # permute the text_emb and temp_emb
        text_emb=text_emb.permute(0, 2, 1, 3)    # [b, c, l, d]
        temp_emb=temp_emb.permute(0, 2, 1, 3)    # [b, c, n, d]

        # reshape the text_emb and temp_emb
        text_emb=text_emb.reshape(B*C, L_text, D_text)
        temp_emb=temp_emb.reshape(B*C, L_temp, D_temp)

        # cross attention
        result = self.cross_encoder(tgt=text_emb, memory=temp_emb) # [b*c, l, d] text as query
```

The process:
1. Reshapes to `[B*C, L, D]` — flattens across channels
2. Cross-attention runs per channel: channel i’s text attends only to channel i’s time series
3. Reshapes back to `[B, C, L, D]`

So there is no global text representation that all channels attend to; it’s channel-specific.

Implications for mmitransformer:
- In mmitransformer, text embeddings become additional variates, so all variates (time series and text) can attend to each other. This is more flexible than TGTSF’s channel-specific setup.
- The channel dimension from `text_encoder` may not be necessary if we want global text or a different number of text variates.
- We can pool over channels (e.g., mean) to get a single global text variate, or keep multiple text variates as a hyperparameter.

This supports making the number of text variates a hyperparameter and potentially pooling over channels when we want fewer text variates than time series channels.

Does TGTSF have any attention between time series variates at all? It does a patching mechanism, and time series representations have attention along the patch dimension, but do they have attention in the cross-sectional (between variates) dimension?

Checking how TGTSF processes time series and whether there's cross-variate attention:


[2 tools called]


Looking at the `TS_encoder` code:

```220:241:layers/TGTSF_torch.py
    def forward(self, x):
        B, L, C = x.shape
        x = x.permute(0, 2, 1)
        x = x.unfold(-1, self.patch_len, self.stride) # [B, C, patch_num, patch_len]

        x = self.input_encoder(x) # [B, C, patch_num, embedding_dim]

        x = rearrange(x, 'b c n d -> (b c) n d', b=B, c=C)

        # print(x.shape, self.W_pos.shape)

        x = x + self.W_pos.permute(1, 0, 2)

        x = self.dropout_layer(x)

        # attn_mask = torch.triu(torch.full((x.shape[1], x.shape[1]), -np.inf), 1).to(x.device)

        x = self.attentions(x)

        x = rearrange(x, '(b c) n d -> b n c d', b=B, c=C)

        return x
```

**No, TGTSF does not have attention between time series variates.**

What happens:
1. Input: `[B, L, C]` → patches → `[B, C, patch_num, embedding_dim]`
2. Reshape: `[B, C, patch_num, embedding_dim]` → `[B*C, patch_num, embedding_dim]` (flattens across channels for batch processing)
3. Attention: `self.attentions(x)` runs on `[B*C, patch_num, embedding_dim]` — attention is only along the patch dimension (temporal), not across channels
4. Reshape back: `[B, patch_num, C, embedding_dim]`

Each channel is processed independently. The reshape to `B*C` is for batch efficiency; attention operates within each channel’s patches.

Then in the mixer:
- Text embeddings: `[B, L_text, C, D_text]`
- Time series: `[B, patch_num, C, D_temp]`
- Reshaped to `[B*C, ...]` for per-channel cross-attention

So TGTSF has:
- ✅ Attention along patches (temporal) within each variate
- ✅ Cross-attention between text and time series (per variate)
- ❌ No attention between different variates

This differs from iTransformer, which attends across variates. In mmitransformer, by treating text as additional variates, we get cross-variate attention, which TGTSF does not have.
