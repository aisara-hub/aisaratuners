from setuptools import setup, find_packages

classifiers = [
    'Development Status :: Production/Stable',
    'Intended Audience :: AI experts :: keras users',
    'Operating System :: OS Independent',
    'License :: OSI Approved :: Wizardry License With Selling Exception',
    'Programming Language :: Python :: 3'
]

setup(
    name='aisaratuners',
    version='0.1.0',
    description='leveraging aisara algorithm for effective hyperparameter tuning',
    long_description=open('README.md').read() + '\n\n' + open('CHANGELOG.txt').read(),
    url='https://github.com/aisara-hub/aisaratuners',
    author='AiSara Artificial Intelligence',
    author_email='devs@aisara.ai',
    license='Wizardry License With Selling Exception',
    classifiers=classifiers,
    python_requires = '3.6',
    packages=find_packages(exclude = ['tests','data','docs']),
    install_requires=['requests', 'pandas', 'numpy','os','json','itertools','random','getpass','plotly', 'tensorflow','math']
)