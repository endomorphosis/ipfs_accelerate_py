from setuptools import setup, find_packages

setup(
	name='ipfs_accelerate_py',
	version='0.0.27',
    packages=find_packages(),
	install_requires=[
		'ipfs_kit_py',
		'sentence_transformers'
        'transformers',
        'ipfs_transformers_py',
        'ipfs_model_manager_py',
		'torch',
        'torchvision',
        'numpy',
        'torchtext',
		'urllib3',
		'requests',
		'boto3',
		'trio',
		'InstructorEmbedding',
		'FlagEmbedding',
		'llama-cpp-python',
		'gguf',
		"optimum",
		"optimum[openvino]",
	]
)