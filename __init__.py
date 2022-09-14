try:
    from setup import setup_environ, download_weights
except:
    from .setup import setup_environ, download_weights

setup_environ()
download_weights()

