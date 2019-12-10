from distutils.core import setup
setup(
  name = 'diver',         # How you named your package folder (MyLib)
  packages = ['diver'],   # Chose the same as "name"
  version = '0.0.0',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'diver is a series of tools to speed up common feature-set investigation, conditioning and encoding for common ML algorithms',   # Give a short description about your library
  author = 'Tom Walker',                   # Type in your name
  author_email = 'tom.walker@clearscore.com',      # Type in your E-Mail
  url = 'https://github.com/ClearScore/diver',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/ClearScore/disc/archive/v_0.0.0.tar.gz',    # I explain this later on
  keywords = ['dataset', 'conditioner', 'feature engineering', 'data inspection'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'pandas',
          'numpy',
          'sklearn',
          'datetime',
          'joblib',
          'seaborn',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)

