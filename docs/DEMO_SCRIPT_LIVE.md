# Live demo script (no slides needed)

A demo that runs entirely in the app: everything the audience sees is a page of the live app, so it works with any slides or none. It runs about **7:45** at a normal pace (≈140 words a minute plus clicks). That leaves 2 minutes of slack under a 10-minute limit.

**App:** <https://which-earns-the-homepage.streamlit.app/>

Speakers are **A** (opening and close), **B** (the four questions) and **C** (Phase 2 and the decision). For two presenters, B also takes C's part. For one, drop the hand-off lines.

## At a glance

| Time | Who | Page | What the reviewers see |
|---|---|---|---|
| 0:00–1:15 | A | Feature Picks (top) | The client, the Black Friday problem and the mandate |
| 1:15–3:15 | B | Feature Picks | The quadrant, the verdict and the live AI (Q1, Q2) |
| 3:15–3:45 | B | Holiday Shift | Black Friday changes conversion, unevenly (Q3) |
| 3:45–4:15 | B | Device Split | One line-up works on both devices (Q4) |
| 4:15–5:00 | C | Outside Signal | Phase 2: demand outside the store |
| 5:00–5:20 | C | Cannibalization Check | Phase 2: swaps don't steal sales |
| 5:20–6:40 | C | Slot Plan | The decision, live edit, AI write-up, export |
| 6:40–7:45 | A | Data & Method | Real data, honest limits, what we'd fix before shipping |
| 7:45–10:00 | all | | Slack, then Q&A |

## Before you start (10 minutes early)

1. Open the app and wait for the first page. It sleeps when idle, so the first load takes 20–30 s.
2. In the sidebar, open **AI narrator**, paste the OpenRouter key and pick **anthropic/claude-haiku-5.5**.
3. Click **Write AI verdict** and **Write AI notes** once as a dry run. The results are cached, so they come back fast live.
4. Open one tab per page in the order of the table (links at the bottom). Zoom the browser to 110–125%.
5. Start the local app as a backup: `.venv\Scripts\python -m streamlit run app.py`. It runs from the committed data, so it works offline; only the AI buttons need the network.
6. Put Feature Picks on screen, scrolled to the top.

## The script

Spoken lines are quoted. **Do** lines are actions.

### 1 · Opening: A, Feature Picks top (0:00–1:15)

On screen: the page title, the line "Products ranked by how often a view turns into a purchase, not by traffic", and the KPI row.

> "We're Team 9. Our client is the Head of Merchandising at the Google Merchandise Store. Every holiday season they decide which few products get the homepage's feature slots for Black Friday, Cyber Monday and the December gifting push.
>
> That decision matters more every year. Last Black Friday, US shoppers spent a record 11.8 billion dollars online, but they placed fewer orders and bought fewer items per order: the growth came from higher prices. Stores had one of their softest Black Friday weekends in recent years. More people looking, each one buying less. So a homepage slot can't go to whatever gets looked at. Every view has to convert."

**Do:** point at the **Forgone purchases** tile (5,277).

> "Today the slots tend to follow traffic. In this store's own data, products with lots of views but below-average conversion lost about 5,277 purchases over the season."

**Do:** point at the line under the title.

> "So our mandate is to rank products by how often a view turns into a purchase, not by traffic. [B] will show you how."

### 2 · Feature Picks: B (1:15–3:15)

**Do:** scroll to the scatter.

> "Each bubble is a product. Across is product views, up is purchases per view. Both axes are log scales, because views run from 200 to 41,000; on a linear axis the small products would crush into one corner. The dashed lines split the products into four groups. Top-left are hidden gems: they convert well, but few people see them. Bottom-right is wasted exposure: lots of views, few sales."

**Do:** scroll up to the **Verdict** card.

> "The app reads the chart for you: feature the Google Keychain and two campus mugs, and retire the Campus Bike Eco Tee. It had 41,040 views at 0.10 percent, about 192 lost purchases."

**Do:** sidebar › **AI narrator** › model **Claude Haiku 5.5** › **Write AI verdict**.

> "That was the rule-based version. This is the AI version: the same numbers, written as a recommendation you could read out in a meeting. It takes about five seconds and costs a tenth of a cent."

**Do:** turn on **Point out what matters** › **Write AI notes**.

> "The AI can also point at what matters on the chart. It can only point at products the app offers it, and every number it writes is checked against the data. A note that fails is dropped."

**Do:** click the **G1** bubble, or **Inspect** under its note.

> "And we show our caveats. The Keychain converts at 20.7 percent, but it's flagged views under-counted, so we'd test it before giving it the lead slot."

### 3 · Holiday Shift: B (3:15–3:45)

**Do:** open Holiday Shift and scroll to the category bars.

> "Question three: does Black Friday change the picks? Conversion rises from 0.51 to 0.79 percent in Cyber Week, but not evenly. Bags more than double; Electronics doesn't move. Each bar has a 95 percent interval, and a percent label appears only when the change is significant, so nobody chases noise."

### 4 · Device Split: B (3:45–4:15)

**Do:** open Device Split.

> "Question four: mobile versus desktop. Both convert at 0.57 percent. Each cell compares a product with its own usual rate, so colour shows a lean, not size. It's mostly gray: none of the top 20 products leans to one device, so one homepage line-up works for both. [C] will take the opportunity."

### 5 · Outside Signal: C (4:15–5:00)

**Do:** open Outside Signal; show the alert, then scroll to the small multiples.

> "This is our Phase 2, beyond the brief: trending outside our walls. We pulled Google search interest for 13 keywords and mapped them to the store's categories. Both lines are indexed to 100, because Google rescales every keyword, so we compare change, not level. Lifestyle and Gifts stands out: search interest tripled into December, 3.07 times, while our own views grew only 1.24 times. Demand outside is outrunning our exposure, so the app lists the best converters to feature before the rush: Keychain, Lapel Pin, Magnet."

### 6 · Cannibalization Check: C (5:00–5:20)

**Do:** open Cannibalization Check.

> "The second Phase 2 feature checks whether featuring one product steals sales from another. Each product is compared with its own pace on nearby days. Nothing reaches 'likely'; the two pairs flagged here are only 'possible', and neither is in the plan."

### 7 · Slot Plan: C (5:20–6:40)

**Do:** open Slot Plan; start on the Black Friday column.

> "And here's the decision: four slots each for Black Friday, Cyber Monday and December. Black Friday leads with the Keychain and the campus mugs. Each card shows a score, an 80 percent range and badges. Black Friday weekend is only four days, so most of each score comes from the full season, and the card says so: mostly season evidence."

**Do:** **Exclude (e.g. out of stock)** › Google Keychain.

> "Say the Keychain is out of stock. Exclude it, and the plan refills: the Lapel Pin moves up into Black Friday."

**Do:** **Write AI verdict** (the plan's verdict, under the cards).

> "The AI writes the plan up for the meeting."

**Do:** **Download plan (Markdown)**.

> "And it exports, so the decision leaves the app. [A] will close."

### 8 · Close: A, Data & Method (6:40–7:45)

**Do:** open Data & Method; show **Source: the data loads, and both routes agree**.

> "Quickly, why you can trust it: this is Google's real GA4 sample, pulled live from BigQuery. That's 4,295,584 events, 270,154 users and 92 days, the same counts as our pitch."

**Do:** scroll to **What this data can't tell us**.

> "And we're upfront about its limits. So, before we ship this to the client, three things. One: the top gems look under-counted, so we'd A/B test them in one slot before they lead a campaign. Two: connect stock and margin. Today the out-of-stock exclude is manual, and with prices rising, margin matters more. Three: rerun it on this year's data. The pipeline runs as is, and mobile is now 55 percent of Black Friday online sales, so we'd re-check the device split.
>
> In short: the homepage goes to products that convert, not just products that get looked at. Thank you. Questions?"

## If something breaks

| Problem | Do this |
|---|---|
| Hosted app asleep or down | Switch to the local app tab: same pages, same data, and it works offline. |
| AI call slow or failing | The app shows the rule-based verdict and notes with a warning. Say: "That's the designed fallback: same numbers, no AI." Move on. |
| Running long | Cut Cannibalization Check to its one line, then skip the Inspect click. |
| Running short | On Holiday Shift, scroll to **Which products change quadrant?** (69 products moved). |

## Q&A prep

**"Wasn't 2025 a record Black Friday?"** Yes, in dollars ($11.8B online, Adobe). But orders fell 1%, items per order fell 2% and prices rose 7% (Salesforce), so the growth is price, not volume.

**"Your data is from 2020. Does it still apply?"** 2020 is when Black Friday moved online (+21.6%, Adobe). The method isn't tied to that year; the pipeline reruns on a current GA4 export.

**"Is the Keychain really at 20.7%?"** Probably not that high. It has 67 purchases on 324 views, but 370 checkout events list it, so its views are under-counted. The app flags it, shrinks it to 16.0%, and we'd test it first.

**"How do you know featuring causes sales?"** We don't: the data has no placement history. The app recommends candidates, and an A/B test confirms them.

**"What if the AI makes something up?"** It can only point at anchors the app offers. Every number must round from that anchor's data, and a failing note is dropped and named. Without a key, the notes are rule-based.

**"Why not join views to purchases on item_id?"** In this sample, ids match purchases to views only 1.1% of the time; product names match 93.2%.

**"Black Friday is only four days. Is that enough data?"** No, so the score blends the window with the product's full-season rate. Each card shows an 80% range and says "Mostly season evidence" when the window carries less than half.

**"Why conversion and not profit?"** The sample has no margin or cost data, only revenue.

## What each rubric category gets

| Rubric category | Where |
|---|---|
| Problem Motivation | Opening: the client, record-dollars-fewer-items, 5,277 purchases lost |
| The Mandate | Feature Picks: the quadrant and verdict; Slot Plan: the decision |
| The Opportunity | Outside Signal (3.07× vs 1.24×) and Cannibalization Check |
| Visual Grammar | Said out loud on each page: log scales, 95% intervals, colour vs each product's own rate, indexed to 100 |
| AI Integration | Live AI verdict and notes on Feature Picks, the AI write-up of the Slot Plan, and the checks it passes |
| Professional Delivery | One path through the app, clean hand-offs (A → B → C → A), under 8 minutes |
| Partner critique | The close names the fixes first: A/B test the gems, stock and margin, current data |

## Page links

Feature Picks is the default page, so it is the bare URL (`/feature_picks` shows "Page not found").

- Feature Picks: <https://which-earns-the-homepage.streamlit.app/>
- Holiday Shift: <https://which-earns-the-homepage.streamlit.app/holiday_shift>
- Device Split: <https://which-earns-the-homepage.streamlit.app/device_split>
- Outside Signal: <https://which-earns-the-homepage.streamlit.app/outside_signal>
- Cannibalization Check: <https://which-earns-the-homepage.streamlit.app/cannibalization>
- Slot Plan: <https://which-earns-the-homepage.streamlit.app/slot_plan>
- Data & Method: <https://which-earns-the-homepage.streamlit.app/data_method>

The Black Friday figures and their sources are listed at the end of [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md#black-friday-sources-slide-2).
