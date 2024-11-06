from setuptools import setup

setup(
	name='ipfs_accelerate_py',
	version='0.0.12',
	packages=[
        'ipfs_accelerate_py'
	],
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