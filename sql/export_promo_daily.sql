-- Export: on-site promotion impressions/clicks by day, device, promotion and item (rule R4).
WITH pr AS (
  SELECT
    PARSE_DATE('%Y%m%d', e.event_date) AS event_date,
    LOWER(IFNULL(e.device.category, 'unknown')) AS device_category,
    e.event_name,
    IF(TRIM(it.promotion_id) IN ('', '(not set)', '<Other>', '(data deleted)', 'Not available in demo dataset'), NULL, TRIM(it.promotion_id)) AS promotion_id,
    IF(TRIM(it.promotion_name) IN ('', '(not set)', '<Other>', '(data deleted)', 'Not available in demo dataset'), NULL, TRIM(it.promotion_name)) AS promotion_name,
    IF(TRIM(it.creative_name) IN ('', '(not set)', '<Other>', '(data deleted)', 'Not available in demo dataset'), NULL, TRIM(it.creative_name)) AS creative_name,
    IF(TRIM(it.creative_slot) IN ('', '(not set)', '<Other>', '(data deleted)', 'Not available in demo dataset'), NULL, TRIM(it.creative_slot)) AS creative_slot,
    IF(TRIM(it.item_id) IN ('', '(not set)', '<Other>', '(data deleted)', 'Not available in demo dataset'), NULL, TRIM(it.item_id)) AS item_id,
    IF(TRIM(it.item_name) IN ('', '(not set)', '<Other>', '(data deleted)', 'Not available in demo dataset'), NULL, TRIM(it.item_name)) AS item_name,
    FARM_FINGERPRINT(CONCAT(IFNULL(e.user_pseudo_id, ''), '|', CAST(e.event_timestamp AS STRING), '|', e.event_name)) AS event_key
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` AS e
  LEFT JOIN UNNEST(e.items) AS it
  WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
    AND e.event_name IN ('view_promotion', 'select_promotion')
)
SELECT
  event_date,
  device_category,
  event_name,
  promotion_id,
  promotion_name,
  creative_name,
  creative_slot,
  item_id,
  item_name,
  COUNT(*) AS item_rows,
  COUNT(DISTINCT event_key) AS events
FROM pr
GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9
