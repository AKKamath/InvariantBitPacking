TESTS = ./tests
BUILD = ./build
NCU=/home/x_kamathak/miniforge3/envs/dgl-dev-gpu-117/bin/../nsight-compute/2022.2.0/ncu
DLRM ?= ./dlrm_feats
OUTPUT = ./results

${OUTPUT}:
	mkdir -p ${OUTPUT}

copy_test: ${BUILD}/copy_test ${OUTPUT}
	./${BUILD}/copy_test > ${OUTPUT}/output.txt
	python scripts/plot.py ${OUTPUT}/output.txt ${OUTPUT}/output

${BUILD}/copy_test: ${TESTS}/copy_test.cu
	nvcc -o $@ $< -I ./include -arch=sm_70 -Xcompiler -fopenmp

decompress_pcie:
	${NCU} --kernel-name "decompress_fetch_cpu_kernel" --metrics pcie__read_bytes.sum.per_second python tests/decompression_thput.py 1

sens_thresh: ${OUTPUT}
	python tests/sens_threshold.py > ${OUTPUT}/sens_thresh.out
	python scripts/get_threshold.py ${OUTPUT}/sens_thresh.out "1 2 4 8" >> ${OUTPUT}/sens_thresh.out

sens_sweep: ${OUTPUT}
	python tests/sens_sweep.py "pubmed citeseer cora reddit products" > ${OUTPUT}/sens_sweep.out
	python tests/sens_sweep.py "mag" > ${OUTPUT}/sens_sweep_mag.out

dlrm: ${DLRM}/feature_0_part0.npy ${DLRM}/feature_1_part0.npy ${DLRM}/feature_2_part0.npy ${DLRM}/feature_3_part0.npy \
	${DLRM}/feature_4_part0.npy ${DLRM}/feature_5_part0.npy ${DLRM}/feature_6_part0.npy ${DLRM}/feature_7_part0.npy ${DLRM}/feature_8_part0.npy \
	${DLRM}/feature_9_part0.npy ${DLRM}/feature_10_part0.npy ${DLRM}/feature_11_part0.npy ${DLRM}/feature_12_part0.npy ${DLRM}/feature_13_part0.npy \
	${DLRM}/feature_14_part0.npy ${DLRM}/feature_15_part0.npy ${DLRM}/feature_16_part0.npy ${DLRM}/feature_17_part0.npy ${DLRM}/feature_18_part0.npy \
	${DLRM}/feature_19_part0.npy ${DLRM}/feature_20_part0.npy ${DLRM}/feature_21_part0.npy ${DLRM}/feature_22_part0.npy ${DLRM}/feature_23_part0.npy \
	${DLRM}/feature_24_part0.npy ${DLRM}/feature_25_part0.npy
	python tests/compression_dlrm.py

${DLRM}/feature_%_part0.npy:
	mkdir -p ${DLRM}
	wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/dle/dlrm_base_tf2_ckpt_ds-criteo-fl15/22.06_tf32/files?redirect=true&path=feature_$*_part0.npy' -O ${DLRM}/feature_$*_part0.npy