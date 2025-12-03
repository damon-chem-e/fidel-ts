# Overall

Separate into text models, vision models, and models that incorporate both. There are datasets that are rich, and those that are not rich. I think for each dataset we should measure the dataset richness in terms of timesteps, timesteps with another modality, number of variates. The number of timesteps per entity is also important in panel settings. It's also important to note that a 'panel' in the sense that I consider for modeling must have time syncronization across entities, which things like TTC medical does not have. Those are independent not cross-dependent channels in that setting (not an interdependent panel). I think that some methods (like MMTSFlib) will be performant on low data settings, and my work will be performant in data rich settings.


Note: I ignore datasets in which the alternate modality is static (not time varying). I note these datasets, and note why I think they should be ignored.

Note: for datasets that are not data rich, but are intermingled time and alternate modality, I want to show that my methods are less performant unless pretrained on data rich datasets in the same domain. LLM prompting is better for simple one-shot tasks about context (CiK), and perhaps on data low settings (Time-MMD, TTC), but my methods are better for data rich settings with complex interdependencies. This is the conjecture.

## Summary

Use data rich datasets for training and evaluation: Amazon reviews, my financial dataset(s), Fidel-TS, and LEMMA-RCA. Also benchmark on others (Time-MMD, TTC, NewsForecast, MTBench weather) but with a focus on showing how methods differ in quality based on complexity and richness of data.

Vision (apart from SEED-DV) is backburner for now. Maintain progress on SEED-DV and test new algos on it. We can parse Image-EEG and amazon reviews plus VISUELLE later if useful.


## Text 

### Microservices
- LEMMA-RCA (text logs + second level time series, over 1 TB total data, > 100 million text log events and > 100 million time stamps for metrics on average per fault, and loads of faults). This should be a nice goldmine.

### Retail 
- Amazon reviews (review text + review frequency + review stars)

### Cross-domain
- Time-MMD (not data rich, note that it already 'preprocesses' text a bit)
- TTC (not data rich, climate + medical)

### Electric Load and Weather
- Fidel-TS (fantastic data rich high quality weather with text)
- NewsForecast (electric load + news articles in Australia, NeurIPS 2024)
- MTBench (regional weather event descriptions + time series from 50 US stations)

### Finance
- NewsForecast (financial news from yahoo and GDELT + bitcoin and FX daily... I can do better than this. Do my thing below.)
- MTBench (financial news aggregated 2021-2023 from GlobeNews, marketwatch, seekingalpha, zacks, invezz, quartz, pennystocks, benzinga + 3000 stocks daily OHLC, volume, vwap, transaction number) -- this is just a starting point, which will be greatly improved
- FNSPID (this is essentially Benzinga scraped, could and should just use Benzinga API, which is included in improving and expanding MTbench)
- CIKM18 or equivalently StockNet (has a lot of twitter data collected from before it was expensive, not super extensive but worth a shot, 88 stocks 2014-2016)
- My datasets
-- GDELT (full news 15 min increments + US equities 15 minutes + crypto 15 min + macro series fred + oil + FX), GDELT ingests news, parses it to what happened in structured form, but does not redistribute originals. still very valuable I think for world events effects on markets. Very rich data.
-- financial statements + earnings calls (couple hours after in 5 minute increments + several days after maybe a week daily) -- empty text where no news available, train the unimodal on full series (massive) and train the multimodal only using the periods with news available (on the residual of course)
-- Go through the same sources in MTBench and scrape backward through time if possible using wayback machine or whatever needed to get daily backwards as far as possible (will need to clearly note that we don't know the timestamp on that date with these) -- maybe don't need wayback machine just grab still existing URLs like MTbench did, then set up a scraper to continually do this moving forward and keep it running
-- a combination of all of the above with clear headers in text denoting which type of text data it is and what the source is

### Medical
- MIMIC-III and MIMIC-IV (text diagnoses, notes, lab events, prescriptions, procedures, etc all time stamped and for over a decade -- also hourly clinical data per patient for 10 years -- incredibly data rich) very rich but not priority for me. very different domain. the data is also painful and messy to work with due to richness and complexity.

## Vision

### Medical (EEG)
- Image-EEG (images shown in sequence, not videos)
- SEED-DV (videos)

### Retail 
- VISUELLE (product images and sales for an italian online retailer 2016-2019)
- Amazon reviews (product images and review frequency + review stars)

## Vision + Text

### Retail 
- Amazon reviews (product images + review text + review frequency + review stars)


## Ignore

- Terra (geospatial data). Contains text description (location, topography, altitude, land type) and geo-images and time series of rainfall. Data rich (millions of locations and years of data), but the text description is static and could have been encoded differently than text (ie numeric one-hot encoded land type, latitude, longitude, altitude, can just be numeric features). Possibly LLM knows some relationships between the text and the series, but it could just be learned without text encodings. If these varied with time, that would be more interesting.
- CiK is a dataset that is not a long history of intermingled text (or other alternate modality) and time series, but rather a set of short time series (extremely data poor) with text context. My work is not trying to use a foundation text model to understand and reason about a data poor setting in which we want some language model to reason about the text and infer time series from the context, but rather the case in which there is a large set of data, and intuition might be insufficient because there is a mass of intermingled text and time series in a large interdependent panel. Multimodal, yes, but this is a different task setting. Perhaps a foundational model with the right domains in it trained in my methods would do well on CiK, but this is a different problem category.
- NYC taxi, NYC bike. These have static text descriptions. Same reason as Terra to ignore.
- LRS3, VoxCeleb2. These are data rich. They have video which can help predict a time series of waveforms (audio). This is a different problem category (see the reasoning for CiK). It's not a messy interdependent panel of alternate data and noisy time series. 
- TSQA not intermingled text and time series.
- TimeCAP. This one is LLM generated summaries, it's not actually a dataset of intermingled text and time series.
- DOW30. This is just bad. Ignore completely.
- FinBen. This is just bad. Ignore completely.
- PTB-XL. This is time series with static text classification. Wrong problem category.
