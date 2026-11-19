# Demo script: "Which Products Earn the Homepage?"

The run of show for Team 9's 10-minute peer-reviewed demo (Account D, Google Merchandise Store), written for the three presenters. It is built around the six rubric categories reviewers score: Problem Motivation, The Mandate, The Opportunity, Visual Grammar, AI Integration and Professional Delivery. It also covers the "Partner critique".

- **Slides:** <https://claude.ai/artifact/FmNvCfnXgJsrDn9kbqCANr>. It is private until shared from the deck's Share menu. Share › Export gives .pptx or PDF. Each slide's speaker notes hold the talk track below.
- **Live app:** <https://which-earns-the-homepage.streamlit.app/>

Speakers are **A** (opening and recap), **B** (demo part 1) and **C** (demo part 2). Swap in names, or fold B and C into one person for a two-person team.

## Run of show

| Time | Who | What | Rubric category it lands |
|---|---|---|---|
| 0:00–2:30 | A | Slides 1–4: title, Black Friday hook, the client's pain, the mandate | 1 Motivation, 2 Mandate |
| 2:30–5:15 | B | **Live:** Feature Picks → Holiday Shift → Device Split | 2 Mandate, 4 Viz, 5 AI |
| 5:15–7:45 | C | **Live:** Outside Signal → Cannibalization (one sentence) → Slot Plan | 3 Opportunity, 2 Mandate |
| 7:45–9:30 | A | Slides 6–10: Phase 2, visual grammar, AI, the decision, before we ship | 3, 4, 5, Partner critique |
| 9:30–10:00 | all | Buffer, then Q&A | 6 Delivery |

Slide 5 is the demo route; leave it on screen while B switches to the browser. Slides 11–12 are backup screenshots.

## Before the demo (10 minutes early)

1. Open the hosted app and wait for the first page. It sleeps when idle, and the first load takes 20–30 s.
2. In the sidebar, open **AI narrator**, paste the OpenRouter key and pick **anthropic/claude-haiku-5.5** (about $0.001 and 4–7 s per call). Keep about $0.10 of credit on the key.
3. Dry-run **Write AI verdict** and **Write AI notes** once. Results are cached, so they come back fast during the demo.
4. Open one tab per page (the links are below) and zoom the browser to 110–125% for the projector.
5. Start the local app as a backup: `.venv\Scripts\python -m streamlit run app.py`, or on Abel's machine `conda.exe run -n ITCS5122 python -m streamlit run app.py`.
6. Open the deck in Present mode, with the speaker notes on a second screen if possible.

### Page links

Feature Picks is the default page, so it lives at the bare URL. `/feature_picks` shows "Page not found".

| Page | Link |
|---|---|
| Feature Picks | <https://which-earns-the-homepage.streamlit.app/> |
| Holiday Shift | <https://which-earns-the-homepage.streamlit.app/holiday_shift> |
| Device Split | <https://which-earns-the-homepage.streamlit.app/device_split> |
| Outside Signal | <https://which-earns-the-homepage.streamlit.app/outside_signal> |
| Cannibalization Check | <https://which-earns-the-homepage.streamlit.app/cannibalization> |
| Slot Plan | <https://which-earns-the-homepage.streamlit.app/slot_plan> |

## Talk track

About 140 words a minute. Text in quotes is said aloud; text in brackets is an action.

### A: slides 1–4 (2:30)

**1 · Title (15 s).** "Good afternoon, we're Team 9. Our client is the Head of Merchandising at the Google Merchandise Store. Every holiday they make one decision with real money on it: which few products get the homepage's feature slots for Black Friday, Cyber Monday and the December gifting push. Our app makes that decision. First, why it matters more now."

**2 · Black Friday: record dollars, fewer items (40 s).** "Last Black Friday, US shoppers spent a record 11.8 billion dollars online, according to Adobe. Sounds great. But underneath it, Salesforce saw online orders fall 1 percent and items per order fall 2 percent, while prices rose 7 percent. Store traffic fell almost 6 percent over the weekend: RetailNext called it one of the softest Black Friday weekends in recent years. More people shopped than ever, and each one bought less. The growth is prices, not volume. Our data is from 2020, the year Black Friday moved online. For a merchandiser the lesson is simple: a homepage slot can't go to whatever gets looked at. Every view has to convert."

**3 · A handful of slots, chosen by traffic (30 s).** "Today the slots tend to go to the products with the most traffic, and that has a measurable cost. Products that get lots of views but convert below the 0.57 percent store average lose about 5,277 purchases over the season. The worst single case is the Campus Bike Eco Tee: 41,040 views at 0.10 percent, roughly 192 lost purchases. And it bites hardest exactly when the slots are chosen: in Cyber Week purchases per day doubled, while views rose only 30 percent."

**4 · Rank by efficiency, not traffic (30 s).** "So the mandate: rank products by view-to-purchase efficiency, not traffic. Conversion is purchases over product views, adjusted toward the category average so three sales on ten views can't win on luck. Here's what the app finds for our four questions. The data is real: 4.3 million GA4 events pulled live from BigQuery, matching our pitch. [B] will show you the app."

### B: live demo, part 1 (2:45)

**Feature Picks (~90 s)**
- "Each bubble is a product: views across, conversion up, both on log scales because views run from 200 to 41,000. Top-left are hidden gems, bottom-right is wasted exposure."
- Read the first line of the **Verdict** card.
- [Sidebar › AI narrator › model **Claude Haiku 5.5** › **Write AI verdict**] "Same numbers, written as a recommendation for a meeting: about 5 seconds, a tenth of a cent."
- [Turn on **Point out what matters** › **Write AI notes**]
- [Click the G1 bubble, or **Inspect** under its note] "Google Keychain: note the *views under-counted* flag. We show our caveats."

**Holiday Shift (~40 s).** "Conversion goes from 0.51 to 0.79 percent in Cyber Week, but not evenly: Bags more than double, Electronics doesn't move. The whiskers are 95 percent intervals, and a percent label only appears when the change is significant."

**Device Split (~30 s).** "Desktop and mobile both convert at 0.57 percent. Each cell compares a product with its own usual rate: mostly gray, so none of the top 20 leans to one device. One homepage line-up works for both." Hand over to C.

### C: live demo, part 2 (2:30)

**Outside Signal (~40 s).** "This is Phase 2. Google search interest in Lifestyle and Gifts tripled into December, 3.07 times, while our own views grew 1.24 times. Outside demand is outrunning our exposure, so the app lists the best converters to promote ahead of it: Keychain, Lapel Pin, Magnet."

**Cannibalization (~15 s, one sentence).** "We also check whether featuring one product steals from another. Once each peer is compared with its own pace, no pair is likely; the two possible ones aren't in the plan."

**Slot Plan (~80 s)**
- Start with the Black Friday column. "Four slots each for Black Friday, Cyber Monday and December. Each card has a score, an 80 percent range and badges: mostly season evidence, converts on both devices, trending on Google, views under-counted."
- [**Exclude (e.g. out of stock)** › Google Keychain] "Say it's out of stock." The Lapel Pin moves up into Black Friday, and the later campaigns refill.
- [**Write AI verdict** for the plan]
- [**Download plan (Markdown)**] Then switch back to the slides.

### A: slides 6–10 (1:45)

**6 · Two features beyond the brief (20 s).** "Two features go beyond the brief. Trending outside our walls tells the client which categories are about to be in demand before their own traffic shows it. The cannibalization check makes sure a swap doesn't steal from another slot, and we caught our own early false alarm."

**7 · Each chart fits its question (20 s).** "Each chart fits its question. Log scales, because views span 200 to 41,000. Intervals on the bars, so a lift isn't mistaken for noise. The heatmap shows lean, not size. And search interest is indexed to 100, because Google rescales every keyword. Every chart also has a Show data table."

**8 · AI writes the note; the app checks it (25 s).** "AI is a force multiplier here: it turns the scatterplot into a sentence for a meeting, and points at what matters. We don't trust it blindly. It can only point at what the app offers, every number is checked against the data, and a note that fails is dropped. In our last live run, all 24 passed."

**9 · The holiday slot plan (20 s).** "The decision: Black Friday leads with the Keychain and the campus mugs, Cyber Monday with the Lapel Pin, December with the Magnet. The Eco Tee comes off the homepage. The Black Friday and Cyber Monday windows are short, so those picks lean on the full season, and the app says so."

**10 · Before we ship (25 s).** "Before we ship, three fixes. Test the top gems with a one-slot A/B test, because their views look under-counted. Connect stock and margin: with prices up 7 percent, margin matters more. And rerun on this year's data, since mobile now drives 55 percent of Black Friday online sales. Thank you, we're happy to take questions."

## If something breaks

| Problem | Do this |
|---|---|
| Hosted app asleep or down | Switch to the local app tab. Same pages, same data. |
| AI call slow or failing | The app falls back to the rule-based verdict and notes with a warning. Say: "that's the designed fallback: same numbers, no AI." |
| No network at all | Use backup slides 11–12, and the screenshots on slides 7 and 8. |
| Running long | Drop Cannibalization in the demo, then shorten slide 7 to one sentence. |

## Q&A prep

**"Wasn't 2025 a record Black Friday?"** Yes, in dollars: $11.8B online (Adobe). But orders fell 1%, items per order fell 2% and prices rose 7% (Salesforce), so the growth is price, not volume. The figures are not adjusted for inflation.

**"Your data is from 2020. Does it still apply?"** 2020 is the year Black Friday moved online (Adobe: +21.6%), and online has kept growing since. The method is not tied to 2020: the pipeline reruns on a current GA4 export, and that's fix 3 on our last slide.

**"Is Google Keychain really converting at 20.7%?"** Probably not that high. It has 67 purchases on 324 product views, but 370 checkout events list it, so shoppers reach it without a product page and its views are under-counted. The app flags it, the adjusted rate is 16.0%, and we'd A/B test it before giving it a lead slot.

**"Why not join views to purchases on item_id?"** In this obfuscated sample, 1,374 ids map to 429 product names. A purchased item matches a viewed item by id only 1.1% of the time, but by name 93.2%.

**"How do you know featuring a product causes sales?"** We don't: there is no homepage-placement history in the data. The app recommends candidates; a one-slot A/B test is how the client confirms them.

**"Why log scales on the scatter?"** Views run from about 200 to 41,040. On a linear axis every hidden gem would sit in one corner.

**"What if the AI makes something up?"** It can only point at anchors the app offers (product codes, category bars, product-device cells). Every number in its text must be a rounding of a number in that anchor's data. A note that fails is dropped, and the page says which and why. Without a key, the app writes rule-based notes from the same numbers.

**"Why have a cannibalization toggle if no pair is likely?"** Two pairs are "possible" (p < 0.05, q = 0.29 after correction): Android Iconic Pin with Google Sunglasses, and with Noogler Android Figure. Neither is in the plan; the toggle keeps such pairs apart if one is pinned in.

**"Should mobile get its own homepage?"** Not on this data. Desktop and mobile both convert at 0.57%, and none of the top 20 candidates converts significantly better on one device. With mobile now at 55% of Black Friday online sales, we'd re-check this on current data.

**"Black Friday weekend is only 4 days. Isn't that too little data?"** Yes, so the score blends the window's own purchases with the product's full-season rate, adjusted by how its category moved in that window. Each card shows an 80% range and says "Mostly season evidence" when less than half the score comes from the window.

**"Why conversion and not profit?"** The sample has no margin or cost data, only revenue. Connecting margin is fix 2.

**"How much does the AI cost?"** Claude Haiku 5.5 costs about $0.001 and 4–7 s per verdict or chart's notes (measured). Claude Opus 5.5 costs about $0.02–0.05 per verdict.

**"How would the client run this next season?"** Rerun the pipeline scripts against their GA4 BigQuery export. Every query is dry-run first and byte-capped; the whole build used about 7 GiB of BigQuery's free 1 TiB a month.

**"How do you know it's right?"** There are automated tests on every push, including synthetic data with planted hidden gems, wasted products, device leans and a cannibal pair, which the metrics must recover. `verify_snapshots.py` also checks the data against the pitch's counts.

## Rubric map: where each category is shown

| Rubric category | Where it's shown | The sentence a reviewer can quote |
|---|---|---|
| 1 Problem Motivation | Slides 2–3 | "Record crowds, smaller baskets: every homepage view has to convert." The client is the Head of Merchandising; 5,277 purchases are lost to high-traffic, low-converting products. |
| 2 The Mandate | Slide 4, Feature Picks quadrant, Slot Plan | "Rank by efficiency, not traffic": the quadrant ranks 368 products by purchases per view, and the Slot Plan turns that into 4 picks per campaign. |
| 3 The Opportunity | Outside Signal, slide 6 | Lifestyle & Gifts search interest grew 3.07× vs 1.24× on-site, so promote Keychain, Lapel Pin and Magnet ahead of demand. The cannibalization check keeps swaps safe. |
| 4 Visual Grammar | Every page, slide 7 | Log-scale quadrant, bars with 95% intervals, a heatmap against each product's own rate, and lines indexed to 100. |
| 5 AI Integration | Feature Picks and Slot Plan AI buttons, slide 8 | The AI verdict turns the chart into a meeting-ready recommendation; every number is checked, and failing notes are dropped. |
| 6 Professional Delivery | The whole run | Problem → mandate → live proof → opportunity → decision → what we'd fix, with clean hand-offs (A → B → C → A). |
| Partner critique | Slide 10 | We name the top fix ourselves: A/B test the under-counted gems before they lead a slot. |

## Black Friday sources (slide 2)

All figures are US, nominal dollars.

| Figure | Source |
|---|---|
| 2025: $11.8B online, +9.1%; 55.2% from mobile | [Adobe 2025 holiday recap](https://business.adobe.com/assets/pdfs/resources/sdk/adobe-holiday-shopping-recap/2025-holiday-shopping-recap.pdf) |
| 2025 Cyber Monday: $14.25B, +7.1% | [Adobe release](https://news.adobe.com/news/2025/12/adobe-cyber-monday-hits-record) |
| 2025: orders −1%, items per order −2%, average selling price +7% | Salesforce, via [CNN](https://www.cnn.com/2025/12/01/business/black-friday-shoppers-holiday-season) and [Chain Store Age](https://chainstoreage.com/ai-mobile-discounts-help-drive-online-black-friday-sales-higher) |
| 2025: store traffic −5.8% Fri–Mon, "one of the softest Black Friday weekends in recent years" | [RetailNext](https://retailnext.net/press-release/retailnext-reports-5-8-dip-in-black-friday-weekend-traffic) |
| 2025: a record 202.9M weekend shoppers | [NRF](https://nrf.com/media-center/press-releases/thanksgiving-holiday-weekend-draws-a-record-203-million-shoppers) |
| 2024: $10.8B, +10.2% | [ABC News](https://abc13.com/post/black-friday-shoppers-spent-record-108-billion-online-what-did-people-buy/15614368/) |
| 2023: $9.8B, +7.5% | [Digital Commerce 360](https://www.digitalcommerce360.com/2023/11/25/black-friday-online-sales-2023/) |
| 2022: $9.12B, +2.3% | [GPB / NPR](https://www.gpb.org/news/2022/11/26/black-friday-sales-rake-in-record-912-billion-online-shoppers) |
| 2021: $8.9B, −1.3%, the first-ever decline | [Practical Ecommerce](https://www.practicalecommerce.com/sales-report-2021-thanksgiving-day-black-friday-cyber-monday) (−1.3% vs $9B; Digital Commerce 360 gives −1.4% vs $9.03B), [Retail Brew](https://retailbrew.com/stories/2021/11/29/black-friday-2021-by-the-numbers-online-sales-dip-slightly-to-usd8-9-billion) ("for the first time ever") |
| 2020: $9.0B, +21.6%; smartphones $3.6B, 40% | [TechCrunch](https://techcrunch.com/2020/11/28/black-friday-online-sales-numbers/) |
