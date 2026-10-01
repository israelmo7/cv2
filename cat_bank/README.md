# Cat Bank

Live webcam app that detects cats with YOLO, enrolls them by name while streaming, and recognizes them later.

## How it works

1. **Detect** — YOLOv8 finds cats in the live frame  
2. **Enroll** — press `e`, type a name, save several crops into the bank  
3. **Recognize** — new crops are matched with ResNet18 embeddings + cosine similarity  

Unknown cats are labeled `Unknown`.

## Setup

```bash
cd cat_bank
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

First run downloads YOLO weights (`yolov8n.pt`) and ResNet18 weights.

## Run (webcam)

```bash
python main.py
# or:
python main.py --source 0
```

### Controls

| Key | Action |
|-----|--------|
| `e` | Start enrollment for the primary (largest) cat |
| type + Enter | Set the cat's name, then auto-capture crops |
| Esc | Cancel naming |
| `d` | Toggle boxes / labels |
| `l` | List cats in the bank |
| `q` / `x` | Quit |

### Other sources

```bash
# Image (useful for testing)
python main.py --source path/to/cat.jpg

# Video file
python main.py --source path/to/video.mp4
```

## Cat bank layout

```
cats_bank/
  index.json
  milo/
    001.jpg
    002.jpg
  luna/
    001.jpg
```

Enrollment happens **while streaming** — no separate photo upload step.

## Tips for better ID

- Enroll 8+ crops (default) with the cat facing different angles  
- Keep lighting similar between enrollment and recognition  
- Raise `--match-threshold` (default `0.85`) if names get mixed up  
- Lower it (e.g. `0.78`) if true matches show as `Unknown`  
- `--match-margin` (default `0.04`) rejects IDs when two enrolled cats score too close

## Headless smoke test

```bash
python smoke_test.py
```

Downloads a sample cat image, enrolls it as `SmokeCat`, and checks that recognition returns that name.

## Notes

- COCO YOLO only detects the class **cat** — dogs and other animals are ignored  
- Very similar-looking cats can still be confused; more enrollment angles help  
- Scale and rotation are handled much better than the old OpenCV template matcher
