from setuptools import setup

setup(
	name='ipfs_accelerate',
	version='0.0.1',
	packages=[
        'ipfs_accelerate',
	],
	install_requires=[
        'transformers',
        'torch',
        'torchvision',
        'numpy',
        'torchtext',
		'urllib3',
		'requests',
		'boto3',
	]
)