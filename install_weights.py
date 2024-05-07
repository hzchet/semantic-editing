import os
import gdown


if __name__ == '__main__':
    stylegan_url = 'https://drive.google.com/uc?id=1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT'
    stylegan_output_dir = 'saved/stylegan2/'
    os.makedirs(stylegan_output_dir, exist_ok=True)
    stylegan_output = f'{stylegan_output_dir}/stylegan2-ffhq-config-f.pt'
    gdown.download(stylegan_url, stylegan_output)
