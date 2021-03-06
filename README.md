# Project Annual

## Technologies :

- Python
- Rust
- Angular
- Flask

## Setup :

1. Dependencies :

   - `Python`([Download](https://www.python.org/downloads/release/python-368/))
   - `Rust`([Download](https://www.rust-lang.org/tools/install))
   - `Flask`([pip install Flask](https://flask.palletsprojects.com/en/1.1.x/))
   - `NPM`([Download](https://www.npmjs.com/))

2. Remove all DS store to be sure

```console
$ find . -name '.DS_Store' -type f -delete
```

3. Setup the flask web service :
   First we need to get in the directory of the backend app and run the FLASK WEB SERVICE

```console
$ cd Project/
$ cd App/
$ cd backend/
$ export FLASK_APP=predict_app.py
$ flask run --host=0.0.0.0
```

The Flask web service should be running on **http://localhost:5000** or **http://0.0.0.0:5000/**

4. Setup the angular front end to interact with models and test them:

```console
$ cd Project/
$ cd App/
$ cd frontend/
$ npm install
$ ng serve
```

The Angular front-end app should be running on **http://localhost:4200/**

5. Run tensorboard to view all the logs.

```console
$ tensorboard --logdir Project/Docs/Results/Logs
```

The tensorboard interface should be running on **http://localhost:6006/**
