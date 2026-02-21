TESTS = ./tests
BUILD = ./build
DLRM ?= ./dlrm_feats
GNN_DATASET ?= ./workloads/Legion-IBP/dataset
OUTPUT = ./results
NCU=$(which ncu)
GCC ?= $${CONDA_PREFIX}/bin/gcc
GPP ?= $${CONDA_PREFIX}/bin/g++
NVCC ?= $${CONDA_PREFIX}/bin/nvcc

.PHONY: kvcache dlrm gnn nvcomp_comparison install_miniconda create_env install_cuda \
	install_deps install_nvcomp install_ndzip install_ibp install_legion install_dgl install \
	clean_install download_dlrm copy_test2 decompress_pcie sens_thresh kmeans nvcomp_kvcache \
	nvcomp_dlrm nvcomp_gnn nvcomp_comparison copy_test decomp_thput dlrm llm_layer sens_sweep invariance

# ------------------------ Initial Setup ------------------------ 

${OUTPUT}:
	mkdir -p ${OUTPUT}

install_miniconda:
	# Create conda directory
	mkdir -p ~/miniconda3
	# Download conda to home directory
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
	# Install conda
	bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
	# Init
	~/miniconda3/condabin/conda init

CUDAV ?= 11.7
TORCH_VERSION ?= 1.13.0
create_env:
	cd workloads/DGL-IBP; \
	bash script/create_dev_conda_env.sh -s -g ${CUDAV} -n ibp -t ${TORCH_VERSION};

install_cuda11:
	conda install nvidia/label/cuda-11.7.0::cuda-toolkit -y
	conda install -c conda-forge gcc_linux-64==11.1.0 gcc==11.1.0 gxx==11.1.0 -y

install_cuda12:
	#conda install -c conda-forge cuda-toolkit==12.1.1 -y
	conda install -c nvidia cuda-toolkit=12.1 -y
	conda install -c conda-forge gcc_linux-64==11.1.0 gcc==11.1.0 gxx==11.1.0 -y

install_deps: ${OUTPUT}
	conda install -c conda-forge libxcrypt cmake=3.19 \
		libboost libboost-devel libboost-headers libboost-python libboost-python-devel -y

install_nvcomp:
	# Install NVComp
	conda install -c conda-forge nvcomp==3.0.1 -y

install_ndzip:
	# Install NDZip
	cd ndzip; \
	cmake -B build -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CXX_FLAGS="-march=native" -DCMAKE_CUDA_COMPILER="${NVCC}" \
		-DCMAKE_C_COMPILER="${GCC}" -DCMAKE_CXX_COMPILER="${GPP}" \
		-DCMAKE_CUDA_HOST_COMPILER="${GPP}"; \
	cmake --build build -j

install_ibp:
	# Install IBP
	pip install -v . --no-build-isolation

install_legion:
	# Install Legion
	cd workloads/Legion-IBP; \
	bash build.sh

install_dgl:
	cd workloads/DGL-IBP; \
	bash script/build_dgl.sh -g; \
	cd python; \
	python setup.py install; \
	python setup.py build_ext --inplace

install_colossalai:
	cd workloads/ColossalAI; \
	pip install -v -e .

create_infinigen:
	conda create -n infinigen python=3.9 -y
	cd workloads/InfiniGen-IBP; \
	conda run -n infinigen pip install -r requirements.txt;

install_infinigen:
	cd workloads/InfiniGen-IBP/speedup; \
	sh install.sh;

install: ${OUTPUT}
	$(MAKE) install_nvcomp
	$(MAKE) install_ndzip
	$(MAKE) install_ibp
	$(MAKE) install_legion
	$(MAKE) install_dgl
	$(MAKE) install_colossalai

clean_install:
	rm -rf build;
	cd ndzip; rm -rf build;
	cd workloads/Legion-IBP/sampling_server; rm -rf build;
	cd workloads/DGL-IBP; rm -rf build;
	cd workloads/DGL-IBP/tensoradapter/pytorch; rm -rf build;
	cd workloads/DGL-IBP/graphbolt/; rm -rf build;

# ------------------------ Dataset Download ------------------------ 

${DLRM}/feature_%_part0.npy:
	mkdir -p ${DLRM}
	wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/dle/dlrm_base_tf2_ckpt_ds-criteo-fl15/22.06_tf32/files?redirect=true&path=feature_$*_part0.npy' -O ${DLRM}/feature_$*_part0.npy

${DLRM}/asteroid.f32:
	mkdir -p ${DLRM}
	wget https://dps.uibk.ac.at/~fabian/sc21-reproducibility/datasets/asteroid.f32.zst -o ${DLRM}/asteroid.f32.zst
	unzstd ${DLRM}/asteroid.f32.zst

download_dlrm: ${DLRM}/feature_0_part0.npy ${DLRM}/feature_1_part0.npy ${DLRM}/feature_2_part0.npy ${DLRM}/feature_3_part0.npy \
	${DLRM}/feature_4_part0.npy ${DLRM}/feature_5_part0.npy ${DLRM}/feature_6_part0.npy ${DLRM}/feature_7_part0.npy ${DLRM}/feature_8_part0.npy \
	${DLRM}/feature_9_part0.npy ${DLRM}/feature_10_part0.npy ${DLRM}/feature_11_part0.npy ${DLRM}/feature_12_part0.npy ${DLRM}/feature_13_part0.npy \
	${DLRM}/feature_14_part0.npy ${DLRM}/feature_15_part0.npy ${DLRM}/feature_16_part0.npy ${DLRM}/feature_17_part0.npy ${DLRM}/feature_18_part0.npy \
	${DLRM}/feature_19_part0.npy ${DLRM}/feature_20_part0.npy ${DLRM}/feature_21_part0.npy ${DLRM}/feature_22_part0.npy ${DLRM}/feature_23_part0.npy \
	${DLRM}/feature_24_part0.npy ${DLRM}/feature_25_part0.npy

${GNN_DATASET}/%/features:
	cd ${GNN_DATASET}/.. && bash prepare_dataset_generic.sh $*

download_gnn: ${GNN_DATASET}/pubmed/features ${GNN_DATASET}/citeseer/features ${GNN_DATASET}/cora/features \
	${GNN_DATASET}/reddit/features ${GNN_DATASET}/products/features ${GNN_DATASET}/mag/features

llm_data.zip:
	wget https://zenodo.org/records/18572812/files/Archive.zip?download=1 -O llm_data.zip

download_llm: llm_data.zip
	mkdir -p kvcache/
	unzip llm_data.zip -d kvcache
	rm -r kvcache/__MACOSX/

# ------------------------ Tests ------------------------ 

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

kmeans: ${DLRM}/asteroid.f32 ${OUTPUT}
	python tests/kmeans_asteroid.py > ${OUTPUT}/kmeans.out
	python scripts/plot_kmeans.py ${OUTPUT}/kmeans.out ${OUTPUT}/kmeans

# PLOTTED EXPERIMENTS
nvcomp_kvcache:
	#python tests/nvcomp_comparison.py kvcache 0 > ${OUTPUT}/kv_wiki_inst.log
	python tests/nvcomp_comparison.py kvcache 1 > ${OUTPUT}/kv_wiki_plaintiff.log

nvcomp_dlrm: ${DLRM}/feature_0_part0.npy ${DLRM}/feature_1_part0.npy ${DLRM}/feature_2_part0.npy ${DLRM}/feature_3_part0.npy \
	${DLRM}/feature_4_part0.npy ${DLRM}/feature_5_part0.npy ${DLRM}/feature_6_part0.npy ${DLRM}/feature_7_part0.npy ${DLRM}/feature_8_part0.npy \
	${DLRM}/feature_9_part0.npy ${DLRM}/feature_10_part0.npy ${DLRM}/feature_11_part0.npy ${DLRM}/feature_12_part0.npy ${DLRM}/feature_13_part0.npy \
	${DLRM}/feature_14_part0.npy ${DLRM}/feature_15_part0.npy ${DLRM}/feature_16_part0.npy ${DLRM}/feature_17_part0.npy ${DLRM}/feature_18_part0.npy \
	${DLRM}/feature_19_part0.npy ${DLRM}/feature_20_part0.npy ${DLRM}/feature_21_part0.npy ${DLRM}/feature_22_part0.npy ${DLRM}/feature_23_part0.npy \
	${DLRM}/feature_24_part0.npy ${DLRM}/feature_25_part0.npy
	python tests/nvcomp_comparison.py dlrm > ${OUTPUT}/dlrm.log

nvcomp_gnn: download_gnn
	for i in pubmed citeseer cora reddit products; do \
		python tests/nvcomp_comparison.py $${i} > ${OUTPUT}/$${i}.log; \
	done

nvcomp_gnn_mag: download_gnn
	python tests/nvcomp_comparison.py mag > ${OUTPUT}/mag.log
	python scripts/extract_compression.py ${OUTPUT} "pubmed citeseer cora reddit products mag dlrm kv_wiki_plaintiff" > ${OUTPUT}/nvcomp_comparison.log

nvcomp_comparison: # Tables 1, 2
	$(MAKE) nvcomp_kvcache
	$(MAKE) nvcomp_dlrm
	$(MAKE) nvcomp_gnn
	python scripts/extract_compression.py ${OUTPUT} "pubmed citeseer cora reddit products dlrm kv_wiki_plaintiff" > ${OUTPUT}/nvcomp_comparison.log

copy_test: ${BUILD}/copy_test.exe ${OUTPUT} # Figure 5
	./${BUILD}/copy_test.exe > ${OUTPUT}/output.txt
	python scripts/plot.py ${OUTPUT}/output.txt ${OUTPUT}/output

decomp_thput: # Figure 7
	python tests/decompression_thput2.py > ${OUTPUT}/decomp_thput.out
	tail -n 4 ${OUTPUT}/decomp_thput.out

RUN_MPS = export CUDA_MPS_PIPE_DIRECTORY=$$(pwd)/tmp; \
	export CUDA_MPS_LOG_DIRECTORY=$$(pwd)/tmp; \
	nvidia-cuda-mps-control -d

KILL_MPS =	echo quit | nvidia-cuda-mps-control

# Figure 8
gnn: ${OUTPUT} ${GNN_DATASET}/pubmed/features ${GNN_DATASET}/citeseer/features ${GNN_DATASET}/cora/features \
	${GNN_DATASET}/reddit/features ${GNN_DATASET}/products/features ${GNN_DATASET}/mag/features
	${RUN_MPS}; \
	cd workloads/Legion-IBP; make run_expts_singlegpu; \
	make -s extract_expts_single > ../../${OUTPUT}/gnn_perf.log; \
	${KILL_MPS};

gnn_mag: ${OUTPUT} ${GNN_DATASET}/mag/features
	${RUN_MPS}; \
	cd workloads/Legion-IBP; make run_mag_all_singlegpu; \
	make -s extract_expts_single > ../../${OUTPUT}/gnn_perf.log; \
	${KILL_MPS};

# Figure 9
dlrm: ${DLRM}/feature_0_part0.npy ${DLRM}/feature_1_part0.npy ${DLRM}/feature_2_part0.npy ${DLRM}/feature_3_part0.npy \
	${DLRM}/feature_4_part0.npy ${DLRM}/feature_5_part0.npy ${DLRM}/feature_6_part0.npy ${DLRM}/feature_7_part0.npy ${DLRM}/feature_8_part0.npy \
	${DLRM}/feature_9_part0.npy ${DLRM}/feature_10_part0.npy ${DLRM}/feature_11_part0.npy ${DLRM}/feature_12_part0.npy ${DLRM}/feature_13_part0.npy \
	${DLRM}/feature_14_part0.npy ${DLRM}/feature_15_part0.npy ${DLRM}/feature_16_part0.npy ${DLRM}/feature_17_part0.npy ${DLRM}/feature_18_part0.npy \
	${DLRM}/feature_19_part0.npy ${DLRM}/feature_20_part0.npy ${DLRM}/feature_21_part0.npy ${DLRM}/feature_22_part0.npy ${DLRM}/feature_23_part0.npy \
	${DLRM}/feature_24_part0.npy ${DLRM}/feature_25_part0.npy ${OUTPUT}
	python tests/dlrm_comp_merged.py > ${OUTPUT}/dlrm_comp_merged.out
	tail -n 8 ${OUTPUT}/dlrm_comp_merged.out > ${OUTPUT}/dlrm_perf.log

# Figure 10
llm: ${OUTPUT}
	cd workloads/InfiniGen-IBP; conda run -n infinigen $(MAKE) run_expt > ../../${OUTPUT}/llm_latency.log;

llm_layer: ${OUTPUT}
	$(MAKE) nvcomp_kvcache
	python scripts/extract_layer_comp.py ${OUTPUT} "kv_wiki_plaintiff kv_wiki_inst"

sens_sweep: ${OUTPUT}
	python tests/sens_sweep.py "pubmed citeseer cora reddit products" > ${OUTPUT}/sens_sweep.out
	python tests/sens_sweep.py "dlrm" > ${OUTPUT}/sens_sweep_dlrm.out
	python tests/sens_sweep.py kvcache 0 > ${OUTPUT}/sens_sweep_kvcache.out
	python tests/sens_sweep.py "mag" > ${OUTPUT}/sens_sweep_mag.out

invariance: ${OUTPUT} # Table 3
	python tests/invariance_perc.py "pubmed citeseer cora reddit products mag dlrm" > ${OUTPUT}/invariance.out

# ------------------------ SLURM for tests ------------------------ 

slurm_gnn: ${OUTPUT}
	sbatch scripts/sbatch_gnn.sh

slurm_gnn_mag: ${OUTPUT}
	sbatch scripts/sbatch_gnn_mag.sh

slurm_dlrm: ${OUTPUT}
	sbatch scripts/sbatch_dlrm.sh

slurm_llm: ${OUTPUT}
	sbatch scripts/sbatch_llm.sh

slurm_nvcomp: ${OUTPUT}
	sbatch scripts/sbatch_comparison.sh

slurm_nvcomp_mag: ${OUTPUT}
	sbatch scripts/sbatch_comparison_mag.sh

slurm: ${OUTPUT}
	$(MAKE) slurm_gnn
	$(MAKE) slurm_gnn_mag
	$(MAKE) slurm_dlrm
	$(MAKE) slurm_llm
	$(MAKE) slurm_nvcomp
	$(MAKE) slurm_nvcomp_mag