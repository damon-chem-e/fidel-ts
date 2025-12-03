# Benchmarks

We will use Fidel-TS as the base benchmark including data and benchmark models. It benchmarks against unimodal models including iTransformer, and multimodal models including GPT4TS. It still needs several benchmark models.

## Overall plan

This is an overall plan for comprehensiveness, but we will not focus on benchmarking beyond Fidel-TS at the beginning stages (ie ICML 2026). Beyond that, possibly PR with Fidel-TS and see if they want to collaborate on a more comprehensive benchmark that is cleaner and more modern.

I think similarly to the data note, we want to add to the literature how different methods perform on different datasets (richness, complexity, panel structure, number of covariates, static vs time varying text, etc.). That is something that is missing from the literature.

Note that most of the literature works with static text rather than interleaved text (because of a lack of good benchmarks, which Fidel-TS is trying to address, and Time-MMD did not address very well). So, most of the methods we can ignore for now. We are in a different ball game than a lot of these models, which is a good thing, but when writing we need to be careful and note which we need to compare and which we do not. 

## Unimodal additions

- Toto 
- MOIRAI + MOIRAI MoE
- Chronos 2
- Ti-Rex
- tiny-tsm
- TEMPO

## Multimodal additions

- MMiTransformer
- TimeCMA (text generated from time series via prompts)
- SE-LLM (only uses static text)
- MMTSFlib 

## Ignore

- TsLLM (strange)

## Other unimodal

There are other unimodal things I'd like to benchmark against that aren't necessary common in this area of the literature, but I think perhaps should be.

- Autogluon-TS tuned 4 hours
- PFN-TS
- RealMLP
- TabICL
