import os
import urllib.request
import tarfile
import zipfile

psplib_dir = '/Users/madanibezoui/Documents/Projects/CSP_for_Sched/benchmarks_real/psplib'
os.makedirs(psplib_dir, exist_ok=True)

urls = [
    'https://www.om-db.wi.tum.de/psplib/files/j30.sm.tgz',
    'https://www.om-db.wi.tum.de/psplib/files/j60.sm.tgz',
    'https://www.om-db.wi.tum.de/psplib/files/j30opt.sm'
]

for url in urls:
    filename = os.path.join(psplib_dir, os.path.basename(url))
    print(f"Downloading {url} to {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        if filename.endswith('.tgz') or filename.endswith('.tar.gz'):
            print(f"Extracting {filename}...")
            with tarfile.open(filename, 'r:gz') as tar:
                tar.extractall(path=psplib_dir)
            os.remove(filename)
    except Exception as e:
        print(f"Error processing {url}: {e}")

print("Download and extraction complete.")
