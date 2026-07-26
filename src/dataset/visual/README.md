# DThinkVLN Reward Dashboard

## Run

```bash
python3 src/dataset/visual/app.py \
  --host 0.0.0.0 \
  --port 18091 \
  --log-dir runs/DThinkVLN-P-7B-GRPO-16-sample/reward_logs
```

Open: `http://127.0.0.1:18091`

## Structure

- `app.py`: HTTP server + API endpoint (`/api/data`) + static asset serving.
- `log_loader.py`: JSONL loader, rank parsing, cache signature.
- `transforms.py`: reusable transformations (rolling mean/std, histogram, downsample).
- `chart_specs.py`: chart builders + dashboard assembly.
- `assets/`: frontend (`index.html`, `style.css`, `app.js`).

## Add New Charts Fast

1. Add extraction/transformation logic in `chart_specs.py` (reuse `transforms.py`).
2. Create `build_xxx_chart(...) -> Dict[str, Any]` that returns chart spec:

```python
{
  "id": "chart_id",
  "title": "Chart Title",
  "description": "...",
  "kind": "line" or "bar",
  "x_label": "...",
  "y_label": "...",
  "options": {...},
  "series": [
    {"name": "series_a", "points": [[x1, y1], [x2, y2]], "smoothable": True}
  ]
}
```

3. Append this builder in `build_dashboard(...)->charts`.

## Notes

- Frontend supports:
  - `log_dir` override.
  - rank filtering.
  - smoothing window for line charts.
- Backend supports non-trivial computed charts (mean/variance/window aggregation).
