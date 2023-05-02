from app import app
# do some production specific things to the app
# app.config['DEBUG'] = False

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)