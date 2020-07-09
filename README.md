# Project Annual

## Technologies :

- Python
- Rust

## Setup :

1. Dependencies :

   - `Python`([Download](https://www.python.org/downloads/release/python-368/))
   - `Rust`([Download](https://www.rust-lang.org/tools/install))
   - `git`
   - `Flask`([pip install Flask](https://flask.palletsprojects.com/en/1.1.x/))
   - `NPM`([Download](https://www.npmjs.com/))

2. Setup the flask web service :
   First we need to get in the directory of the backend app and run the FLASK WEB SERVICE

```console
$ cd Project/
$ cd App/
$ cd backend/
$ export FLASK_APP=predict_app.py
$ flask run --host=0.0.0.0
```

The Flask web service should be running on **http://localhost:5000** or **http://0.0.0.0:5000/**

3. Setup the angular front end to interact with models and test them:

```console
$ cd Project/
$ cd App/
$ cd frontend/
$ npm install
$ ng serve
```

The Angular front-end app should be running on **http://localhost:4200/**
