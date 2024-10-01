from setuptools import setup, find_packages

setup(
    name='hybrid_cnn',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        # Add other dependencies like torchvision if needed
    ],
    description='A custom hybrid CNN with residual and SE blocks',
    author='Ranjithkumar',
    author_email='ranjithbca27@gmail.com',
    url='https://https://github.com/ran0707/hybrid_cnn',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
