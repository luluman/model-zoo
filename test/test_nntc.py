import logging
import pytest
import re

def container_run(nntc_env, cmd):
    container = nntc_env['nntc_container']
    logging.info(cmd)
    ret, output = container.exec_run(
        f'bash -c "{cmd}"',
        tty=True)
    output = output.decode()
    logging.info(f'------>\n{output}')
    m = re.search('(?<=please check ).+\\.log', output)
    if m:
        log_fn = m.group(0).replace('/workspace', '.')
        cmd_fn = log_fn.replace('.log', '.cmd')
        with open(cmd_fn) as f:
            logging.info(f'cat {cmd_fn}\n{f.read()}')
        with open(log_fn) as f:
            logging.info(f'cat {log_fn}\n{f.read()}')

    assert ret == 0

@pytest.fixture(scope='module')
def make_lmdb(nntc_env):
    if not nntc_env['case_list']:
        logging.info(f'Skip nntc make lmdb')
        return
    cmd = 'pip3 install -r /workspace/requirements.txt'
    container = nntc_env['nntc_container']
    logging.info(cmd)
    ret, output = container.exec_run(
        f'bash -c "{cmd}"',
        tty=True)
    if ret != 0:
        output = output.decode()
        logging.error(f'------>\n{output}')
    assert ret == 0
    container_run(nntc_env, f'python3 -m tpu_perf.make_lmdb {nntc_env["case_list"]}')

@pytest.mark.parametrize('target', ['BM1684', 'BM1684X'])
def test_accuracy(target, nntc_env, get_imagenet_val, get_cifar100, get_coco2017_val, make_lmdb):
    if not nntc_env['case_list']:
        logging.info(f'Skip nntc accuracy test')
        return

    # Build for accuracy test
    container_run(nntc_env, f'python3 -m tpu_perf.build {nntc_env["case_list"]} --outdir out_acc_{target} --target {target}')

@pytest.mark.parametrize('target', ['BM1684', 'BM1684X'])
def test_efficiency(target, nntc_env, get_imagenet_val, get_cifar100, get_coco2017_val):
    if not nntc_env['case_list']:
        logging.info(f'Skip nntc efficiency test')
        return

    # Build for efficiency test
    container_run(nntc_env, f'python3 -m tpu_perf.build --time {nntc_env["case_list"]} --outdir out_eff_{target} --target {target}')
