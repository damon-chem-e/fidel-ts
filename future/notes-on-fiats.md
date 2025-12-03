Their proposition is basically saying "hey look, if you don't condition on external information (external to historical time series), you get the conditional expectation only conditional on the past time series" which is a tautology in a sense. Basically we need a conditional expectation, but we just need to condition on more things (their 'external stimuli' is just more conditioning information into a conditional expectation).

FIATS is LLM-free, numerical based. Would be great to add into an ensemble.

I think a lot of the FIATS gains are from channel specificity. Small datasets and studies used shared parameters per channel. If you have enough data, learning channel-specific parameters is essential, which FIATS captures. They have channel specificity in boht their text head and in their time series head (once linked with text). They don't really have channel specificity in the ts backbone.

The CASM blocks essentially use KV-Q attention between a channel description and the news itself. Interesting! Again, this requires enough data to learn everything (low data regimes is where LLMs come in handy). This is something I can adapt into my models in the text side. Perhaps alternate this attention block (citing) with self-attention on the news itself. This could be incredibly useful for product description and firm description (channel descriptions), though since I prepend that to news, it's essentially subsumed in my methodology (but mention, and try doing this if my stuff doesn't seem to work very well).

Unfolds the non-overlapping patches into overlapping patches. I think this is cool. Wonder if PatchTST does this.

FIATS also uses full bidirectional attention.

FIATS uses a filtering mechanism to find the samples in which text-guided is useful. That's the point of the gate. They didn't put it in their architecture, but they had the concept there and used it to filter the training samples (seems bad).
