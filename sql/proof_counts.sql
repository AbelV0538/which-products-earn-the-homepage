-- Proof it loads (pitch slide 5, verbatim).
-- Expected: event_count 4295584, user_count 270154, day_count 92.
SELECT
  COUNT(*) AS event_count,
  COUNT(DISTINCT user_pseudo_id) AS user_count,
  COUNT(DISTINCT event_date) AS day_count
FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
