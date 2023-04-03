import logging
import pytest
import subprocess
import utils

@pytest.mark.runtime
@pytest.mark.nntc
def test_nntc_runtime_BM1684X(target, nntc_runtime):
    if not nntc_runtime['case_list']:
        logging.info(f'Skip efficiency test')
        return

    cmd = f'python3 -m tpu_perf.run {nntc_runtime["case_list"]} --outdir out_eff_{target} --target {target}'
    cmd += utils.get_devices_opt()
    logging.info(cmd)
    subprocess.run(cmd, shell=True, check=True)

@pytest.mark.runtime
@pytest.mark.nntc
def test_nntc_precision_BM1684(target, nntc_runtime, get_imagenet_val, get_cifar100, get_coco2017_val):
    if not nntc_runtime['case_list']:
        logging.info(f'Skip precision test')
        return

    cmd = f'python3 -m tpu_perf.precision_benchmark {nntc_runtime["case_list"]} --outdir out_acc_{target} --target {target}'
    cmd += utils.get_devices_opt()
    logging.info(cmd)
    subprocess.run(cmd, shell=True, check=True)
