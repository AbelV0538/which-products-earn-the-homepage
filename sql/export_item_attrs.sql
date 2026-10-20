-- Export: item attribute combinations with counts (canonical name/category resolved in Python).
WITH r AS (
  SELECT
    PARSE_DATE('%Y%m%d', e.event_date) AS event_date,
    e.event_name,
    IF(TRIM(it.item_id) IN ('', '(not set)', '<Other>', '(data deleted)', 'Not available in demo dataset'), NULL, TRIM(it.item_id)) AS item_id,
    IF(TRIM(it.item_name) IN ('', '(not set)', '<Other>', '(data deleted)', 'Not available in demo dataset'), NULL, TRIM(it.item_name)) AS item_name,
    IF(TRIM(it.item_category) IN ('', '(not set)', '<Other>', '(data deleted)', 'Not available in demo dataset'), NULL, TRIM(it.item_category)) AS item_category,
    IF(TRIM(it.item_category2) IN ('', '(not set)', '<Other>', '(data deleted)', 'Not available in demo dataset'), NULL, TRIM(it.item_category2)) AS item_category2,
    IF(TRIM(it.item_brand) IN ('', '(not set)', '<Other>', '(data deleted)', 'Not available in demo dataset'), NULL, TRIM(it.item_brand)) AS item_brand,
    it.price_in_usd
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` AS e, UNNEST(e.items) AS it
  WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
    AND e.event_name IN ('view_item', 'add_to_cart', 'purchase')
)
SELECT
  item_id,
  item_name,
  item_category,
  item_category2,
  item_brand,
  COUNT(*) AS item_rows,
  COUNTIF(event_name = 'view_item') AS view_rows,
  COUNTIF(event_name = 'purchase') AS purchase_rows,
  APPROX_QUANTILES(IF(price_in_usd > 0, price_in_usd, NULL), 2 IGNORE NULLS)[SAFE_OFFSET(1)] AS median_price_usd,
  MIN(event_date) AS first_seen,
  MAX(event_date) AS last_seen
FROM r
WHERE item_id IS NOT NULL OR item_name IS NOT NULL
GROUP BY item_id, item_name, item_category, item_category2, item_brand
