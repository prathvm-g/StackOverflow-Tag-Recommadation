WITH filtered_table AS (
  SELECT
    id,
    title,
    body,
    tags
  FROM `bigquery-public-data.stackoverflow.posts_questions`
  WHERE score >= 10
    AND tags IS NOT NULL
  LIMIT 60000
)

SELECT
  id,
  title,
  body,
  SPLIT(tags, '|') AS tag_list
FROM filtered_table;
