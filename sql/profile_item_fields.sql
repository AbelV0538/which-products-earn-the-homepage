-- Profile: placeholder share and completeness of item fields, per event type.
SELECT
  e.event_name,
  COUNT(*) AS item_rows,
  COUNTIF(IF(TRIM(it.item_id) IN ('', '(not set)', '<Other>', '(data deleted)', 'Not available in demo dataset'), NULL, TRIM(it.item_id)) IS NULL) AS bad_item_id,
  COUNTIF(IF(TRIM(it.item_name) IN ('', '(not set)', '<Other>', '(data deleted)', 'Not available in demo dataset'), NULL, TRIM(it.item_name)) IS NULL) AS bad_item_name,
  COUNTIF(IF(TRIM(it.item_category) IN ('', '(not set)', '<Other>', '(data deleted)', 'Not available in demo dataset'), NULL, TRIM(it.item_category)) IS NULL) AS bad_item_category,
  COUNTIF(it.quantity IS NULL) AS null_quantity,
  COUNTIF(it.price_in_usd IS NULL) AS null_price,
  COUNTIF(it.item_revenue_in_usd IS NULL) AS null_revenue,
  COUNTIF(it.item_revenue_in_usd = 0) AS zero_revenue,
  COUNT(DISTINCT IF(TRIM(it.item_id) IN ('', '(not set)', '<Other>', '(data deleted)', 'Not available in demo dataset'), NULL, TRIM(it.item_id))) AS distinct_ids,
  COUNT(DISTINCT IF(TRIM(it.item_name) IN ('', '(not set)', '<Other>', '(data deleted)', 'Not available in demo dataset'), NULL, TRIM(it.item_name))) AS distinct_names,
  APPROX_TOP_COUNT(it.item_id, 8) AS top_raw_ids,
  APPROX_TOP_COUNT(it.item_name, 8) AS top_raw_names,
  APPROX_TOP_COUNT(it.item_category, 10) AS top_raw_categories,
  APPROX_TOP_COUNT(it.promotion_name, 10) AS top_promotion_names,
  APPROX_TOP_COUNT(it.creative_slot, 10) AS top_creative_slots,
  APPROX_TOP_COUNT(it.item_list_name, 8) AS top_list_names
FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` AS e, UNNEST(e.items) AS it
WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
GROUP BY e.event_name
ORDER BY item_rows DESC
