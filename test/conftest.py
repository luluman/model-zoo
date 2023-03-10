import pytest
import logging
import docker
import os

import re
import io
import tarfile
import requests
from ftplib import FTP
class FTPClient:
    """
    ftp://user_name:password@hostname
    """
    def __init__(self, url):
        prog = re.compile('ftp://(.+)@')
        pat = prog.search(url)
        if pat:
            self.user, self.passwd = pat.group(1).split(':')
        else:
            self.user, self.passwd = None, None
        self.host = prog.sub('', url)

        self.release_type = 'daily_build'
        if os.environ.get('GITHUB_REF', '').endswith('stable'):
            self.release_type = 'release_build'

        self.session = FTP(self.host, user=self.user, passwd=self.passwd)

    def download_and_untar(self, fn):
        logging.info(f'Download & extract {fn}')
        buf = io.BytesIO()
        self.session.retrbinary(
            f'RETR {fn}',
            buf.write)
        buf.seek(0)
        tar = tarfile.open(fileobj=buf)
        tar.extractall()

    def get_nntc(self):
        path = f'/sophon-sdk/tpu-nntc/{self.release_type}/latest_release'
        self.session.cwd(path)
        fn = next(filter(lambda x: x.startswith('tpu-nntc_'), self.session.nlst()))
        logging.info(f'Latest nntc package is {fn}')
        out_dir = fn.replace('.tar.gz', '')
        if os.path.exists(out_dir):
            logging.info(f'{out_dir} already exists')
            return fn
        self.download_and_untar(os.path.join(path, fn))
        return fn

    def get_mlir(self):
        path = f'/sophon-sdk/tpu-mlir/{self.release_type}/latest_release'
        self.session.cwd(path)
        fn = next(filter(lambda x: x.startswith('tpu-mlir_'), self.session.nlst()))
        logging.info(f'Latest mlir package is {fn}')
        out_dir = fn.replace('.tar.gz', '')
        if os.path.exists(out_dir):
            logging.info(f'{out_dir} already exists')
            return fn
        self.download_and_untar(os.path.join(path, fn))
        return fn

from html.parser import HTMLParser

class ReleasePageParser(HTMLParser):
    def __init__(self, *args, **kwargs):
        super(ReleasePageParser, self).__init__(*args, **kwargs)
        self.results = []

    def handle_starttag(self, tag, attrs):
        if tag == 'include-fragment':
            attrs = dict(attrs)
            m = re.match('^.+(\\d+\\.)+\\d+$', attrs.get('src', ''))
            if not m:
                return
            self.results.append(m.group(0))

class ExpandParser(HTMLParser):
    def __init__(self, *args, **kwargs):
        super(ExpandParser, self).__init__(*args, **kwargs)
        self.results = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            attrs = dict(attrs)
            self.results.append(attrs.get('href'))

def get_latest_tpu_perf():
    backoff = 0.5
    url = 'https://github.com/sophgo/tpu-perf/releases'
    for i in range(10):
        try:
            resp = requests.get(url, timeout=15)
            break
        except requests.exceptions.Timeout:
            logging.warning(f'Failed to query {url}, retry after {backoff}s')
            time.sleep(backoff)
            backoff *= 2
    assert resp

    resp.raise_for_status()
    parser = ReleasePageParser()
    parser.feed(resp.text)

    page = parser.results[0]
    backoff = 0.5
    for i in range(10):
        try:
            resp = requests.get(page, timeout=15)
            break
        except requests.exceptions.Timeout:
            logging.warning(f'Failed to query {page}, retry after {backoff}s')
            time.sleep(backoff)
            backoff *= 2
    assert resp

    resp.raise_for_status()
    parser = ExpandParser()
    parser.feed(resp.text)

    return parser.results

tpu_perf_whl = None
@pytest.fixture(scope='session')
def latest_tpu_perf_whl():
    global tpu_perf_whl
    if not tpu_perf_whl:
        tpu_perf_whl = next(filter(lambda x: 'x86' in x, get_latest_tpu_perf()))
    return f'https://github.com/{tpu_perf_whl}'

import shutil
import glob
def remove_tree(path):
    for match in glob.glob(path):
        logging.info(f'Removing {match}')
        shutil.rmtree(match)

@pytest.fixture(scope='session')
def nntc_docker(latest_tpu_perf_whl):
    # Env assertion
    assert os.path.exists('/run/docker.sock')

    root = os.path.dirname(os.path.dirname(__file__))
    logging.info(f'Working dir {root}')
    os.chdir(root)
    remove_tree('out*')

    # Download
    ftp_server = os.environ.get('FTP_SERVER')
    assert ftp_server
    f = FTPClient(ftp_server)
    nntc_fn = f.get_nntc()
    nntc_dir = nntc_fn.replace('.tar.gz', '')

    # Docker init
    client = docker.from_env()
    image = 'sophgo/tpuc_dev:v2.1'
    client.images.pull(image)

    # Glob kernel module
    import glob
    kernel_module = glob.glob(os.path.join(nntc_dir, 'lib/*kernel_module*'))
    assert kernel_module
    kernel_module = kernel_module[0]

    # NNTC container
    nntc_container = client.containers.run(
        image, 'bash',
        volumes=[f'{root}:/workspace'],
        restart_policy={'Name': 'always'},
        environment=[
            f'PATH=/workspace/{nntc_dir}/bin:/usr/local/bin:/usr/bin:/bin',
            f'BMCOMPILER_KERNEL_MODULE_PATH=/workspace/{kernel_module}',
            f'NNTC_TOP=/workspace/{nntc_dir}'],
        tty=True, detach=True)

    if 'GITHUB_ENV' in os.environ:
        with open(os.environ['GITHUB_ENV'], 'a') as f:
            f.write(f'NNTC_CONTAINER={nntc_container.name}\n')

    logging.info(f'Setting up NNTC')
    ret, _ = nntc_container.exec_run(
        f'bash -c "source /workspace/{nntc_dir}/scripts/envsetup.sh"',
        tty=True)
    assert ret == 0

    logging.info(f'NNTC container {nntc_container.name}')

    yield dict(docker=client, nntc_container=nntc_container)

    # Docker cleanup
    logging.info(f'Removing NNTC container {nntc_container.name}')
    nntc_container.remove(v=True, force=True)

    remove_tree('out*')
    remove_tree('data')
    remove_tree('tpu-nntc*')

@pytest.fixture(scope='session')
def mlir_docker(latest_tpu_perf_whl):
    # Env assertion
    assert os.path.exists('/run/docker.sock')

    root = os.path.dirname(os.path.dirname(__file__))
    logging.info(f'Working dir {root}')
    os.chdir(root)
    remove_tree('mlir_out*')

    # Download
    ftp_server = os.environ.get('FTP_SERVER')
    assert ftp_server
    f = FTPClient(ftp_server)
    mlir_fn = f.get_mlir()
    mlir_dir = mlir_fn.replace('.tar.gz', '')
    logging.info(f'mlir_dir: {mlir_dir}')
    # Docker init
    client = docker.from_env()
    image = 'sophgo/tpuc_dev:latest'
    client.images.pull(image)

    # MLIR container
    logging.info(f'Setting up MLIR')
    mlir_container = client.containers.run(
        image, 'bash',
        volumes=[f'{root}:/workspace'],
        restart_policy={'Name': 'always'},
        environment=[
            f'PATH=/workspace/{mlir_dir}/bin:' \
            f'/workspace/{mlir_dir}/python/tools:' \
            f'/workspace/{mlir_dir}/python/utils:' \
            f'/workspace/{mlir_dir}/python/test:' \
            f'/workspace/{mlir_dir}/python/samples:' \
            f'/usr/local/bin:/usr/bin:/bin',
            f'LD_LIBRARY_PATH=/workspace/{mlir_dir}/lib',
            f'PYTHONPATH=/workspace/{mlir_dir}/python'],
        tty=True, detach=True)

    if 'GITHUB_ENV' in os.environ:
        with open(os.environ['GITHUB_ENV'], 'a') as f:
            f.write(f'MLIR_CONTAINER={mlir_container.name}\n')

    logging.info(f'MLIR container {mlir_container.name}')

    yield dict(docker=client, mlir_container=mlir_container)

    # Docker cleanup
    logging.info(f'Removing MLIR container {mlir_container.name}')
    mlir_container.remove(v=True, force=True)

    remove_tree('mlir_out*')
    remove_tree('tpu-mlir*')

import subprocess

def git_commit_id(rev):
    p = subprocess.run(
        f'git rev-parse {rev}',
        shell=True, check=True,
        capture_output=True)
    return p.stdout.decode().strip(' \n')

def git_commit_parents(rev='HEAD'):
    p = subprocess.run(
        f'git rev-parse {rev}^@',
        shell=True, check=True,
        capture_output=True)
    return p.stdout.decode().strip(' \n').split()

def dig(c, callback, depth=0, max_depth=100):
    if not callback(c):
        return
    if depth >= max_depth:
        return
    for p in git_commit_parents(c):
        dig(p, callback, depth + 1, max_depth)

def get_relevant_commits():
    head_parents = git_commit_parents()
    if len(head_parents) == 1:
        return ['HEAD']
    assert len(head_parents) == 2

    base_set = set()
    def cb(x):
        if x in base_set:
            return False
        base_set.add(x)
        return True
    dig(git_commit_id('origin/main'), cb)

    ps = [p for p in head_parents if p not in base_set]
    result = []
    while ps:
        result += ps
        new_ps = []
        for p in ps:
            new_ps += [new_p for new_p in git_commit_parents(p) if new_p not in base_set]
        ps = new_ps

    return result

def git_changed_files(rev):
    p = subprocess.run(
        f'git show --pretty="" --diff-filter=ACMRTUXB --name-only {rev}',
        shell=True, check=True,
        capture_output=True)
    return p.stdout.decode().strip(' \n').split()

from functools import reduce
@pytest.fixture(scope='session')
def case_list():
    if 'TEST_CASES' in os.environ:
        return os.environ['TEST_CASES'].strip() or '--full'

    if os.environ.get('FULL_TEST'):
        return '--full'

    files = reduce(
        lambda acc, x: acc + x,
        [git_changed_files(c) for c in get_relevant_commits()], [])

    # Skip certain files
    files = [
        f for f in files
        if not f.endswith('.md')
        and not f.endswith('.txt')
        and not os.path.basename(f).startswith('.')]

    is_model = lambda x: x.startswith('vision') or x.startswith('language')
    files = [f for f in files if is_model(f)]

    dirs = set([os.path.dirname(f) for f in files])
    def has_config(d):
        try:
            next(filter(lambda x: x.endswith('config.yaml'), os.listdir(d)))
        except StopIteration:
            return False
        else:
            return True
    s = ' '.join(d for d in dirs if has_config(d))
    return s

@pytest.fixture(scope='session')
def nntc_env(nntc_docker, latest_tpu_perf_whl, case_list):
    ret, _ = nntc_docker['nntc_container'].exec_run(
        f'bash -c "pip3 install {latest_tpu_perf_whl}"',
        tty=True)
    assert ret == 0

    logging.info(f'Running cases "{case_list}"')

    yield dict(**nntc_docker, case_list=case_list)

@pytest.fixture(scope='session')
def mlir_env(mlir_docker, latest_tpu_perf_whl, case_list):
    ret, _ = mlir_docker['mlir_container'].exec_run(
        f'bash -c "pip3 install {latest_tpu_perf_whl}"',
        tty=True)
    assert ret == 0

    logging.info(f'Running cases "{case_list}"')

    yield dict(**mlir_docker, case_list=case_list)


def execute_cmd(cmd):
    ret = os.system(cmd)
    assert ret == 0, f'{cmd} failed!'

@pytest.fixture(scope='session')
def get_cifar100():
    data_server = os.environ.get('DATA_SERVER')
    assert data_server
    fn = 'cifar-100-python.tar.gz'

    if len(os.listdir('dataset/CIFAR100/cifar-100-python/')) >= 5:
        logging.info(f'{fn} already downloaded')
    else:
        url = os.path.join(data_server, fn)
        logging.info(f'Downloading {fn}')
        cmd = f'curl -s {url} | tar -zx --strip-components=1 ' \
             '-C dataset/CIFAR100/cifar-100-python/'
        execute_cmd(cmd)

@pytest.fixture(scope='session')
def get_imagenet_val():
    data_server = os.environ.get('DATA_SERVER')
    assert data_server
    fn = 'ILSVRC2012_img_val.tar'
    url = os.path.join(data_server, fn)
    dst = 'dataset/ILSVRC2012/ILSVRC2012_img_val/'
    if len(os.listdir(dst)) >= 50000:
        logging.info(f'{fn} already downloaded')
        return
    logging.info(f'Downloading {fn}')
    cmd = f'curl -s {url} | tar -x -C {dst}'
    execute_cmd(cmd)

@pytest.fixture(scope='session')
def get_coco2017_val():
    data_server = os.environ.get('DATA_SERVER')
    assert data_server

    fn = 'val2017.zip'
    url = os.path.join(data_server, fn)
    if len(os.listdir('dataset/COCO2017/val2017')) >= 5000:
        logging.info(f'{fn} already downloaded')
    else:
        logging.info(f'Downloading {fn}')
        cmd = f'curl -o val2017.zip -s {url}'
        execute_cmd(cmd)
        cmd = 'unzip -o val2017.zip -d dataset/COCO2017'
        execute_cmd(cmd)
        cmd = 'rm val2017.zip'
        execute_cmd(cmd)

    fn = 'annotations_trainval2017.zip'
    if len(os.listdir('dataset/COCO2017/annotations')) >= 7:
        logging.info(f'{fn} already downloaded')
    else:
        url = os.path.join(data_server, fn)
        logging.info(f'Downloading {fn}')
        cmd = f'curl -o annotations.zip -s {url}'
        execute_cmd(cmd)
        cmd = 'unzip -o annotations.zip -d dataset/COCO2017/'
        execute_cmd(cmd)
        cmd = 'rm annotations.zip'
        execute_cmd(cmd)

def main():
    logging.basicConfig(level=logging.INFO)

    files = reduce(
        lambda acc, x: acc + x,
        [git_changed_files(c) for c in get_relevant_commits()], [])
    print(files)

if __name__ == '__main__':
    main()
