-- Export: item x day x device funnel counts.
-- Grain: (event_date, device_category, item_id, item_name) after placeholder cleaning.
-- Exact duplicate events (same user, timestamp, name) are collapsed; purchases are
-- de-duplicated by transaction_id (first event wins; events without an id are kept).
WITH raw AS (
  SELECT
    PARSE_DATE('%Y%m%d', event_date) AS event_date,
    event_name,
    event_timestamp,
    LOWER(IFNULL(device.category, 'unknown')) AS device_category,
    IF(TRIM(ecommerce.transaction_id) IN ('', '(not set)', '<Other>', '(data deleted)', 'Not available in demo dataset'), NULL, TRIM(ecommerce.transaction_id)) AS transaction_id,
    FARM_FINGERPRINT(CONCAT(IFNULL(user_pseudo_id, ''), '|', CAST(event_timestamp AS STRING), '|', event_name)) AS event_key,
    items
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
    AND event_name IN ('view_item', 'add_to_cart', 'begin_checkout', 'purchase', 'view_promotion', 'select_promotion')
),
base AS (
  SELECT *
  FROM raw
  WHERE TRUE
  QUALIFY ROW_NUMBER() OVER (PARTITION BY event_key ORDER BY event_timestamp) = 1
),
first_purchase AS (
  SELECT event_key
  FROM base
  WHERE event_name = 'purchase' AND transaction_id IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (PARTITION BY transaction_id ORDER BY event_timestamp, event_key) = 1
),
events AS (
  SELECT
    b.event_date,
    b.event_name,
    b.device_category,
    b.event_key,
    b.items,
    b.event_name = 'purchase' AND (b.transaction_id IS NULL OR fp.event_key IS NOT NULL) AS counted_purchase,
    b.event_name = 'purchase' AND b.transaction_id IS NULL AS purchase_missing_txn
  FROM base AS b
  LEFT JOIN first_purchase AS fp ON b.event_key = fp.event_key
),
item_rows AS (
  SELECT
    e.event_date,
    e.device_category,
    e.event_name,
    e.event_key,
    e.counted_purchase,
    e.purchase_missing_txn,
    off AS item_offset,
    IF(TRIM(it.item_id) IN ('', '(not set)', '<Other>', '(data deleted)', 'Not available in demo dataset'), NULL, TRIM(it.item_id)) AS item_id,
    IF(TRIM(it.item_name) IN ('', '(not set)', '<Other>', '(data deleted)', 'Not available in demo dataset'), NULL, TRIM(it.item_name)) AS item_name,
    it.quantity,
    it.item_revenue_in_usd
  FROM events AS e, UNNEST(e.items) AS it WITH OFFSET AS off
)
SELECT
  event_date,
  device_category,
  item_id,
  item_name,
  COUNTIF(event_name = 'view_item') AS view_rows,
  COUNT(DISTINCT IF(event_name = 'view_item', event_key, NULL)) AS view_events,
  COUNTIF(event_name = 'view_item' AND item_offset = 0) AS primary_views,
  COUNT(DISTINCT IF(event_name = 'add_to_cart', event_key, NULL)) AS add_to_carts,
  COUNT(DISTINCT IF(event_name = 'begin_checkout', event_key, NULL)) AS checkouts,
  COUNT(DISTINCT IF(counted_purchase, event_key, NULL)) AS purchases,
  COUNT(DISTINCT IF(event_name = 'purchase', event_key, NULL)) AS purchase_events_raw,
  COUNT(DISTINCT IF(purchase_missing_txn, event_key, NULL)) AS purchases_missing_txn,
  IFNULL(SUM(IF(counted_purchase, quantity, NULL)), 0) AS units,
  ROUND(IFNULL(SUM(IF(counted_purchase, item_revenue_in_usd, NULL)), 0), 2) AS revenue_usd,
  COUNT(DISTINCT IF(event_name = 'view_promotion', event_key, NULL)) AS promo_views,
  COUNT(DISTINCT IF(event_name = 'select_promotion', event_key, NULL)) AS promo_clicks
FROM item_rows
WHERE item_id IS NOT NULL OR item_name IS NOT NULL
GROUP BY event_date, device_category, item_id, item_name
