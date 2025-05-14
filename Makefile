TESTS = ./tests
BUILD = ./build
DLRM ?= ./dlrm_feats
OUTPUT = ./results
NCU=$(which ncu)

${OUTPUT}:
	mkdir -p ${OUTPUT}

copy_test2: ${BUILD}/copy_test2.exe ${OUTPUT}
	./${BUILD}/copy_test2.exe > ${OUTPUT}/output.txt
	python scripts/plot.py ${OUTPUT}/output.txt ${OUTPUT}/output

${BUILD}/%.exe: ${TESTS}/%.cu
	nvcc -o $@ $< -I ./include -arch=sm_70 -gencode arch=compute_80,code=sm_80 -Xcompiler -fopenmp -std=c++17

decompress_pcie:
	${NCU} --kernel-name "decompress_fetch_cpu_kernel" --metrics pcie__read_bytes.sum.per_second python tests/decompression_thput.py 1

sens_thresh: ${OUTPUT}
	python tests/sens_threshold.py > ${OUTPUT}/sens_thresh.out
	python scripts/get_threshold.py ${OUTPUT}/sens_thresh.out "1 2 4 8" >> ${OUTPUT}/sens_thresh.out

#dlrm: ${DLRM}/feature_0_part0.npy ${DLRM}/feature_1_part0.npy ${DLRM}/feature_2_part0.npy ${DLRM}/feature_3_part0.npy \
	${DLRM}/feature_4_part0.npy ${DLRM}/feature_5_part0.npy ${DLRM}/feature_6_part0.npy ${DLRM}/feature_7_part0.npy ${DLRM}/feature_8_part0.npy \
	${DLRM}/feature_9_part0.npy ${DLRM}/feature_10_part0.npy ${DLRM}/feature_11_part0.npy ${DLRM}/feature_12_part0.npy ${DLRM}/feature_13_part0.npy \
	${DLRM}/feature_14_part0.npy ${DLRM}/feature_15_part0.npy ${DLRM}/feature_16_part0.npy ${DLRM}/feature_17_part0.npy ${DLRM}/feature_18_part0.npy \
	${DLRM}/feature_19_part0.npy ${DLRM}/feature_20_part0.npy ${DLRM}/feature_21_part0.npy ${DLRM}/feature_22_part0.npy ${DLRM}/feature_23_part0.npy \
	${DLRM}/feature_24_part0.npy ${DLRM}/feature_25_part0.npy
#	python tests/compression_dlrm.py

${DLRM}/feature_%_part0.npy:
	mkdir -p ${DLRM}
	wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/dle/dlrm_base_tf2_ckpt_ds-criteo-fl15/22.06_tf32/files?redirect=true&path=feature_$*_part0.npy' -O ${DLRM}/feature_$*_part0.npy

${DLRM}/asteroid.f32:
	mkdir -p ${DLRM}
	wget https://dps.uibk.ac.at/~fabian/sc21-reproducibility/datasets/asteroid.f32.zst -o ${DLRM}/asteroid.f32.zst
	unzstd ${DLRM}/asteroid.f32.zst

kmeans: ${DLRM}/asteroid.f32 ${OUTPUT}
	python tests/kmeans_asteroid.py > ${OUTPUT}/kmeans.out
	python scripts/plot_kmeans.py ${OUTPUT}/kmeans.out ${OUTPUT}/kmeans

.PHONY: kvcache dlrm gnn nvcomp_comparison

# PLOTTED EXPERIMENTS
nvcomp_kvcache:
	python tests/nvcomp_comparison.py kvcache 0 > ${OUTPUT}/kv_wiki_plaintiff.log
	python tests/nvcomp_comparison.py kvcache 1 > ${OUTPUT}/kv_wiki_inst.log

nvcomp_dlrm: ${DLRM}/feature_0_part0.npy ${DLRM}/feature_1_part0.npy ${DLRM}/feature_2_part0.npy ${DLRM}/feature_3_part0.npy \
	${DLRM}/feature_4_part0.npy ${DLRM}/feature_5_part0.npy ${DLRM}/feature_6_part0.npy ${DLRM}/feature_7_part0.npy ${DLRM}/feature_8_part0.npy \
	${DLRM}/feature_9_part0.npy ${DLRM}/feature_10_part0.npy ${DLRM}/feature_11_part0.npy ${DLRM}/feature_12_part0.npy ${DLRM}/feature_13_part0.npy \
	${DLRM}/feature_14_part0.npy ${DLRM}/feature_15_part0.npy ${DLRM}/feature_16_part0.npy ${DLRM}/feature_17_part0.npy ${DLRM}/feature_18_part0.npy \
	${DLRM}/feature_19_part0.npy ${DLRM}/feature_20_part0.npy ${DLRM}/feature_21_part0.npy ${DLRM}/feature_22_part0.npy ${DLRM}/feature_23_part0.npy \
	${DLRM}/feature_24_part0.npy ${DLRM}/feature_25_part0.npy
	python tests/nvcomp_comparison.py dlrm > ${OUTPUT}/dlrm.log

nvcomp_gnn:
	for i in pubmed citeseer cora reddit products mag paper100m; do \
		python tests/nvcomp_comparison.py $${i} > ${OUTPUT}/$${i}.log; \
	done

nvcomp_comparison: # Tables 1, 2
	$(MAKE) nvcomp_kvcache
	$(MAKE) nvcomp_dlrm
	$(MAKE) nvcomp_gnn
	python scripts/extract_compression.py ${OUTPUT} "pubmed citeseer cora reddit products mag paper100m dlrm kv_wiki_plaintiff" > ${OUTPUT}/nvcomp_comparison.log

copy_test: ${BUILD}/copy_test.exe ${OUTPUT} # Figure 6
	./${BUILD}/copy_test.exe > ${OUTPUT}/output.txt
	python scripts/plot.py ${OUTPUT}/output.txt ${OUTPUT}/output

decomp_thput: # Figure 8
	python tests/decompression_thput2.py

# Figure 10
dlrm: ${DLRM}/feature_0_part0.npy ${DLRM}/feature_1_part0.npy ${DLRM}/feature_2_part0.npy ${DLRM}/feature_3_part0.npy \
	${DLRM}/feature_4_part0.npy ${DLRM}/feature_5_part0.npy ${DLRM}/feature_6_part0.npy ${DLRM}/feature_7_part0.npy ${DLRM}/feature_8_part0.npy \
	${DLRM}/feature_9_part0.npy ${DLRM}/feature_10_part0.npy ${DLRM}/feature_11_part0.npy ${DLRM}/feature_12_part0.npy ${DLRM}/feature_13_part0.npy \
	${DLRM}/feature_14_part0.npy ${DLRM}/feature_15_part0.npy ${DLRM}/feature_16_part0.npy ${DLRM}/feature_17_part0.npy ${DLRM}/feature_18_part0.npy \
	${DLRM}/feature_19_part0.npy ${DLRM}/feature_20_part0.npy ${DLRM}/feature_21_part0.npy ${DLRM}/feature_22_part0.npy ${DLRM}/feature_23_part0.npy \
	${DLRM}/feature_24_part0.npy ${DLRM}/feature_25_part0.npy
	python tests/dlrm_comp_merged.py > ${OUTPUT}/dlrm_comp_merged.out

# Figure 11
llm_layer:
	$(MAKE) nvcomp_kvcache
	python scripts/extract_layer_comp.py ${OUTPUT} "kv_wiki_plaintiff kv_wiki_inst"

sens_sweep: ${OUTPUT}
	python tests/sens_sweep.py "pubmed citeseer cora reddit products" > ${OUTPUT}/sens_sweep.out
	python tests/sens_sweep.py "mag" > ${OUTPUT}/sens_sweep_mag.out
	python tests/sens_sweep.py "dlrm" > ${OUTPUT}/sens_sweep_dlrm.out
	python tests/sens_sweep.py kvcache 0 > ${OUTPUT}/sens_sweep_kvcache.out

invariance: # Table 3
	python tests/invariance_perc.py "pubmed citeseer cora reddit products mag" > ${OUTPUT}/invariance.out