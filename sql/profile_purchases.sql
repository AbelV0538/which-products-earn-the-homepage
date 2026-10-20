-- Profile: purchase events vs transactions (duplicates, missing ids, revenue) (rule R5).
WITH p AS (
  SELECT
    IF(TRIM(ecommerce.transaction_id) IN ('', '(not set)', '<Other>', '(data deleted)', 'Not available in demo dataset'), NULL, TRIM(ecommerce.transaction_id)) AS txn,
    ecommerce.purchase_revenue_in_usd AS rev,
    ARRAY_LENGTH(items) AS n_items
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
    AND event_name = 'purchase'
),
dedup AS (
  SELECT txn, ANY_VALUE(rev) AS rev
  FROM p
  WHERE txn IS NOT NULL
  GROUP BY txn
)
SELECT
  (SELECT COUNT(*) FROM p) AS purchase_events,
  (SELECT COUNTIF(txn IS NULL) FROM p) AS missing_txn,
  (SELECT COUNT(DISTINCT txn) FROM p) AS distinct_txn,
  (SELECT COUNTIF(txn IS NOT NULL) - COUNT(DISTINCT txn) FROM p) AS duplicate_events,
  (SELECT COUNTIF(IFNULL(rev, 0) = 0) FROM p) AS zero_revenue_events,
  (SELECT COUNTIF(n_items = 0) FROM p) AS purchases_without_items,
  (SELECT ROUND(SUM(rev), 2) FROM p) AS purchase_revenue_usd_raw,
  (SELECT ROUND(IFNULL(SUM(rev), 0), 2) FROM dedup)
    + (SELECT ROUND(IFNULL(SUM(rev), 0), 2) FROM p WHERE txn IS NULL) AS purchase_revenue_usd_dedup
