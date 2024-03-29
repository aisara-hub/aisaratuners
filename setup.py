from setuptools import find_packages, setup

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3'
]

setup(
    name='aisaratuners',
    version='1.5.0',
    description='leveraging aisara algorithm for effective hyperparameter tuning',
    license='LICENSE.txt',
    long_description=open('README.md').read() + '\n\n' +
    open('CHANGELOG.txt').read(),
    project_urls={
        'Source Code': 'https://github.com/aisara-hub/aisaratuners',
        'Documentation': 'https://github.com/aisara-hub/aisaratuners/blob/master/docs/user%20guide.md'
    },
    author='AiSara Artificial Intelligence',
    author_email='devs@aisara.ai',
    classifiers=classifiers,
    python_requires='>=3.6',
    packages=find_packages(exclude=['tests', 'data', 'docs']),
    install_requires=['requests', 'pandas', 'numpy',
                      'plotly >= 4.6.0', 'tensorflow >= 1.15']
)
