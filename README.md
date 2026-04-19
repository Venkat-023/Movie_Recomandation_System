# CineMatch AI

CineMatch AI is an upgraded movie recommendation experience built with Streamlit. Instead of a plain checkbox demo, it now feels more like a lightweight streaming discovery product: users can combine genres, start from a favorite movie, and browse polished recommendation cards ranked from a local movie catalog.

## What changed

- Stronger interface with a cinematic visual direction
- Offline-first recommendation flow that removes the fragile IMDb dependency
- Anchor-title mode so users can start from a movie they already love
- Better recommendation cards with match scores and genre reasoning
- Prepared for Hugging Face deployment with Docker and Streamlit config files

## Features

- Genre-based discovery from a catalog of 57k+ movies
- Title-assisted recommendations using cosine similarity
- Fast local inference using pandas and scikit-learn
- Cleaner layout designed for demos, hackathons, and judging
- Deployment-ready structure for Streamlit on Hugging Face Spaces

## Run locally

```bash
pip install -r requirements.txt
streamlit run movie_recomandation_system.py
```

## Project files

- `movie_recomandation_system.py` - main Streamlit app
- `final10.xls` - primary movie dataset used by the recommender
- `requirements.txt` - Python dependencies
- `Dockerfile` - deployment container for Hugging Face Spaces
- `.streamlit/config.toml` - Streamlit runtime configuration

## Deployment

This project has been prepared for Docker-based deployment on Hugging Face Spaces. Once the Hugging Face repository link is connected, the app can be deployed on port `7860`.

### Planned live links

- Hugging Face Space: coming soon
- Direct app URL: coming soon

After deployment, this README will be updated with the live links.

## Recommendation logic

The app loads a local genre matrix from `final10.xls`, converts user genre selections into a preference vector, and ranks movies using cosine similarity. If a user selects an anchor movie, the app blends that movie's genre signature with the selected genres to produce stronger taste-based matches.

## Why this version is better

The previous version relied on live IMDb calls for descriptions and links, which is not ideal for stable hosted deployment. This upgraded version keeps the recommendation flow local and dependable, improves the visual design substantially, and presents the project in a more product-focused way.
