# Snap! Word Explorer

A Streamlit app for children to photograph objects and learn their Korean or English names. It uses YOLOv8n's 80 COCO object classes and provides language-specific audio.
Users can create a username and password, add friends, and start automatic photo battles in beach or forest arenas. Each player chooses a word from their word book when starting or accepting a battle; the latest photo for that word is used. Fighters alternate attacks until one runs out of health. Account data and photos are stored in a local SQLite database; passwords are stored as salted hashes.

## Project structure

```text
.
├── app.py
├── arena.py
├── local_store.py
├── .gitignore
├── requirements.txt
└── README.md
```

The `yolov8n.pt` weights are downloaded automatically by Ultralytics on first use and cached by Streamlit for the running app.

## Run locally

Use Python 3.11, then install dependencies and start Streamlit:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Allow camera access when prompted. Browser speech audio is generated through gTTS, so an internet connection is needed when the audio is first generated.

On first run, the app creates `word_explorer.db` automatically. Users on the same running app can create accounts, search for usernames, accept friend requests, and battle using their newest detected photos. The database file contains account and photo data, is excluded from Git, and should be backed up if it must be preserved.

## Deploy to Streamlit Community Cloud

1. Push `app.py`, `local_store.py`, `requirements.txt`, and `README.md` to the root of a GitHub repository.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) and choose **Create app**.
3. Select the repository, branch, and `app.py` as the main file.
4. Choose Python 3.11 in the advanced settings, then deploy.
5. Open the deployed app on a phone or tablet, grant camera access, and take a photo. The first object detection downloads `yolov8n.pt`; allow time for that initial load.

The SQLite database is local to the app server. Community Cloud files may be reset when the app restarts, so this simple setup does not provide durable cloud storage. Visitors to the public app can create accounts and upload photos; do not use important accounts or personal photos. No server-side camera device or speaker is used. The camera is accessed by the browser through Streamlit, and audio is played in the browser.

The local database, downloaded model weights, Python caches, and Streamlit secrets are excluded from Git. The model weights are downloaded automatically by Ultralytics when the app first runs.
