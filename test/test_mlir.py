import logging
import pytest
import re

def container_run(mlir_env, cmd):
    container = mlir_env['mlir_container']
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

def test_mlir_efficiency(mlir_env):
    if not mlir_env['case_list']:
        logging.info(f'Skip efficiency test')
        return
    container_run(mlir_env, f'python3 -m tpu_perf.build --mlir {mlir_env["case_list"]} --outdir mlir_out')
