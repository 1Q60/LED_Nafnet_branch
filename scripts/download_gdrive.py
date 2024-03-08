import argparse

from led.utils.download_util import download_file_from_google_drive

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--id', type=str, help='File id')
    parser.add_argument('--save-path', type=str, help='Save path')
    args = parser.parse_args()

    download_file_from_google_drive(args.id, args.save_path)
