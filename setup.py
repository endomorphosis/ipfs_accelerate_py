from setuptools import setup, find_packages

setup(
	name='ipfs_accelerate_py',
	version='0.0.16',
    packages=find_packages(),
	install_requires=[
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
	]
)