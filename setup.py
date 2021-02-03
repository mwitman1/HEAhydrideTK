from setuptools import setup

setup(name='heahydridetk',
      version='0.1',
      description='ASE interface for generating HEA alloys and hydride starting configurations for further calculation',
      url='http://github.com/mwitman1/heahydrides',
      author='Matthew Witman',
      author_email='mwitman1@gmail.com',
      license='MIT',
      packages=['heahydridetk'],
      zip_safe=False,
      python_requires='>3.6',
      install_requires=[
        'pymatgen',
        'ase'
      ],
      entry_points={
          'console_scripts': ['heahydridetk=heahydridetk.command_line:main'],
      },
    )

