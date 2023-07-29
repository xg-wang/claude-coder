import hupper
import dotenv

if __name__ == '__main__':
    dotenv.load_dotenv(dotenv.find_dotenv())
    _ = hupper.start_reloader('app.main')

