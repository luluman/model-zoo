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
        with open(log_fn) as f:
            logging.info(f'cat {log_fn}\n{f.read()}')

    assert ret == 0

@pytest.mark.parametrize('target', ['BM1684', 'BM1684X'])
def test_accuracy(target, nntc_env, get_imagenet_val, get_cifar100, get_coco2017_val):
    if not nntc_env['case_list']:
        logging.info(f'Skip nntc accuracy test')
        return

    # Build for efficiency test
    container_run(nntc_env, f'python3 -m tpu_perf.build --time {nntc_env["case_list"]} --outdir out_{target} --target {target}')

    # Build for accuracy test
    container_run(nntc_env, 'pip3 install -r /workspace/requirements.txt')
    container_run(nntc_env, f'python3 -m tpu_perf.make_lmdb {nntc_env["case_list"]}')
    container_run(nntc_env, f'python3 -m tpu_perf.build {nntc_env["case_list"]} --outdir out_{target} --target {target}')
