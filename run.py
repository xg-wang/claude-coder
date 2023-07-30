import hupper
import dotenv

def main():
    dotenv.load_dotenv(dotenv.find_dotenv())
    _ = hupper.start_reloader('src.app.main')

if __name__ == '__main__':
    main()
