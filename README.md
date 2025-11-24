# Conspiracy Mapping Dashboard

This dashboard provides an interactive visualisation of Estonian Facebook and Telegram messages posted from January 2020 to November 2023, focusing on conspiracy-related content. The data was collected from public groups and channels that discuss conspiracy theories.

Messages were processed via natural language processing (NLP) techniques to (a) infer their relevance to conspiracy theory discourse and (b) assign thematic categories. Messages are considered relevant if they fall into one of two categories: 'Conspiracy Theories' (text that promotes, discusses, or alludes to conspiracy theories) or 'Non-Conspiracy Commentary' (text that discusses socially, politically, or culturally relevant topics but without promoting, discussing, or alluding to conspiracy theories).

Relevant messages are further represented in two-dimensions using a dimensionality reduction technique that attempts to preserve the semantic relationships (i.e., two messages that are visually close share similar meanings). A total of 56,290 messages are included in the visualisation.

The visualisation provides an interactive animation of how messaging and the thematic categories changes over time. The animation can be interacted with directly to e.g., zoom into regions, skip to specific time frames, make it fullscreen. Further controls in the sidebar can be used to change the speed, granularity, and category being displayed, and Data Filters can be used to focus in on specific thematic categories, conspiracy theory relevancy, and platforms.

This dashboard was created by [CASM Technology](https://casmtechnology.com/) in collaboration with the [REDACT](https://redactproject.sites.er.kcl.ac.uk/) project. REDACT is supported by the CHANSE ERA-NET Co-fund programme, which has received funding from the European Union’s Horizon 2020 Research and Innovation Programme, under Grant Agreement no. 101004509.

<img width="962" height="709" alt="animation" src="https://github.com/user-attachments/assets/c257558a-efaf-47f6-a001-f3daa75fe9f2" />
