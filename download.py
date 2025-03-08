from audiocaps_download import Downloader
d = Downloader(root_path='audiocaps/', n_jobs=16)
d.download(format = 'wav') # it will cross-check the files with the csv files in the original repository